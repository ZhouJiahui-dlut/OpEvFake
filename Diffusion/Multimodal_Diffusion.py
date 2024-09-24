import torch
import torch.nn as nn
import torch.nn.functional as F

from src.CrossmodalTransformer import MULTModel
from src.StoG import CapsuleSequenceToGraph
from Diffusion.Multimodal_Model import Text_Noise_Pre, Audio_Noise_Pre, Visual_Noise_Pre


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, modelConfig, beta_1, beta_T, T, t_in, a_in, v_in, d_m, dropout, label_dim,
                 unified_size, vertex_num, routing, T_t, T_a, T_v,  batch_size):
        super().__init__()

        self.T = T
        self.batch_size = batch_size
        self.mult_dropout = dropout

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        # Feature Extraction
        self.fc_pre_t_1 = nn.LSTM(t_in, modelConfig["t_in_pre"], bidirectional=True)
        self.fc_pre_t_2 = nn.Linear(modelConfig["t_in_pre"]*2, modelConfig["t_in_pre"])
        self.fc_pre_v = torch.nn.Linear(v_in, modelConfig["v_in_pre"])
        self.fc_pre_com = nn.Sequential(torch.nn.Linear(modelConfig["t_in"], unified_size), torch.nn.ReLU(), nn.Dropout(p=modelConfig["comments_dropout"]))
        self.fc_pre_user = nn.Sequential(torch.nn.Linear(modelConfig["t_in"], unified_size), torch.nn.ReLU(),
                                        nn.Dropout(p=modelConfig["comments_dropout"]))
        self.fc_pre_c3d = torch.nn.Linear(modelConfig["c3d_in"], unified_size)
        self.fc_pre_gpt_1 = nn.LSTM(t_in, modelConfig["t_in_pre"], bidirectional=True)
        self.fc_pre_gpt_2 = nn.Linear(modelConfig["t_in_pre"] * 2, modelConfig["t_in_pre"])

        self.vggish_layer = torch.hub.load(r'/EmotionDiffusion-FakeSV/torchvggish-master', 'vggish', source='local')
        net_structure = list(self.vggish_layer.children())
        self.vggish_modified = nn.Sequential(*net_structure[-2:-1])
        self.fc_pre_a = nn.Linear(a_in, modelConfig["a_in_pre"])

        # Intra-modal Enhancement
        self.fc_g_t = nn.Linear(d_m * 6, d_m)
        self.fc_a_MTout = nn.Linear(d_m * 3, d_m)
        self.fc_v_MTout = nn.Linear(d_m * 3, d_m)
        self.CrossmodalTransformer = MULTModel(modelConfig["t_in_pre"], modelConfig["a_in_pre"], modelConfig["v_in_pre"], d_m, self.mult_dropout)
        self.StoG = CapsuleSequenceToGraph(d_m, unified_size, vertex_num, routing, T_t, T_a, T_v)

        # Cross-modal Interaction
        self.model_t = Text_Noise_Pre(T=modelConfig["T"], ch=modelConfig["vertex_num"],
                           dropout=modelConfig["Text_Pre_dropout"],
                           in_ch=unified_size)
        self.model_a = Audio_Noise_Pre(T=modelConfig["T"], ch=modelConfig["vertex_num"],
                           dropout=modelConfig["Img_Pre_dropout"],
                           in_ch=unified_size)
        self.model_v = Visual_Noise_Pre(T=modelConfig["T"], ch=modelConfig["vertex_num"],
                                       dropout=modelConfig["Img_Pre_dropout"],
                                       in_ch=unified_size)

        self.fc_t = nn.Linear(in_features=vertex_num, out_features=1)
        self.fc_a = nn.Linear(in_features=vertex_num, out_features=1)
        self.fc_v = nn.Linear(in_features=vertex_num, out_features=1)
        self.fc_m = nn.Linear(in_features=unified_size * 3, out_features=unified_size)

        # Prediction
        self.fc_pre = nn.Linear(in_features=unified_size, out_features=label_dim)
        self.trm = nn.TransformerEncoderLayer(d_model=unified_size, nhead=2, batch_first=True)

    def forward(self, texts, audios, videos, comments, c3d, user_intro, gpt_description):
        # Feature Extraction
        texts_local, _ = self.fc_pre_t_1(texts)
        texts_local = self.fc_pre_t_2(texts_local)
        audios = self.vggish_modified(audios)
        audios_local = self.fc_pre_a(audios)
        c3d_local = self.fc_pre_c3d(c3d)
        gpt_local, _ = self.fc_pre_gpt_1(gpt_description)
        gpt_local = self.fc_pre_gpt_2(gpt_local)
        comments_global = self.fc_pre_com(comments)
        user_intro_global = self.fc_pre_user(user_intro.squeeze())
        videos = self.fc_pre_v(videos)
        videos_global = torch.mean(videos, -2)

        # Intra-modal Enhancement
        z_t, z_g, z_a, z_v = self.CrossmodalTransformer(texts_local, gpt_local, audios_local, c3d_local)  # (49,32,64) (200,32,64)
        z_t = self.fc_g_t(torch.cat([z_t, z_g], dim=2))
        z_a = self.fc_a_MTout(z_a)
        z_v = self.fc_v_MTout(z_v)
        x_t, x_a, x_v = self.StoG(z_t, z_a, z_v, self.batch_size) #(32,32,64)

        # Cross-modal Interaction
        x_m = torch.concat([x_t.squeeze(), x_a.squeeze(), x_v.squeeze()], dim=2)
        x_m = self.fc_m(x_m)

        t_t = torch.randint(self.T, size=(x_t.shape[0], ), device=x_t.device) # batchsize (0->T-1)
        noise_t = torch.randn_like(x_t)
        x_tmp_t = (
            extract(self.sqrt_alphas_bar, t_t, x_t.shape) * x_t +
            extract(self.sqrt_one_minus_alphas_bar, t_t, x_t.shape) * noise_t)

        t_a = torch.randint(self.T, size=(x_a.shape[0],), device=x_a.device)
        noise_a = torch.randn_like(x_a)
        x_tmp_a = (
                extract(self.sqrt_alphas_bar, t_a, x_a.shape) * x_a +
                extract(self.sqrt_one_minus_alphas_bar, t_a, x_a.shape) * noise_a)

        t_v = torch.randint(self.T, size=(x_v.shape[0],), device=x_v.device)
        noise_v = torch.randn_like(x_v)
        x_tmp_v = (
                extract(self.sqrt_alphas_bar, t_v, x_v.shape) * x_v +
                extract(self.sqrt_one_minus_alphas_bar, t_v, x_v.shape) * noise_v)

        x_a_pre = self.model_a(x_tmp_a, t_a, x_m)
        x_v_pre = self.model_v(x_tmp_v, t_v, x_m)
        x_t_pre = self.model_t(x_tmp_t, t_t, x_m)
        loss_a = F.mse_loss(x_a_pre.squeeze(), x_a, reduction='none')
        loss_t = F.mse_loss(x_t_pre.squeeze(), x_t, reduction='none')
        loss_v = F.mse_loss(x_v_pre.squeeze(), x_v, reduction='none')
        loss = loss_t + loss_a + loss_v

        output_a = self.fc_a(x_a_pre.transpose(2,1))
        output_t = self.fc_t(x_t_pre.transpose(2,1))
        output_v = self.fc_v(x_v_pre.transpose(2,1))
        output_a = output_a.transpose(2, 1)
        output_t = output_t.transpose(2, 1)
        output_v = output_v.transpose(2, 1)

        comments_global = comments_global.unsqueeze(1)
        videos_global = videos_global.unsqueeze(1)
        user_intro_global = user_intro_global.unsqueeze(1)

        # Prediction
        output_m = torch.concat([output_t, output_a, videos_global, user_intro_global, output_v, comments_global], dim=1)
        output_m = self.trm(output_m)
        output_m = torch.mean(output_m, -2)
        output_m = self.fc_pre(output_m.squeeze())

        return loss, output_m