from typing import Dict
import torch
from tqdm import tqdm

def train(modelConfig: Dict, train_loader, trainer, criterion, optimizer):

    results = []
    truths = []
    trainer.train()
    total_loss = 0.0
    total_batch_size = 0

    for batch in tqdm(train_loader):
        batch_size = batch["label"].size(0)
        texts = batch["text"]
        audios = batch["audioframes"]
        videos = batch["frames"]
        comments = batch["comments"]
        labels = batch["label"]
        c3d = batch["c3d"]
        user_intro = batch["user_intro"]
        gpt_description = batch["gpt_description"]
        total_batch_size += batch_size
        if torch.cuda.is_available():
            audios = audios.cuda()
            texts = texts.cuda()
            videos = videos.cuda()
            comments = comments.cuda()
            labels = labels.cuda()
            c3d = c3d.cuda()
            user_intro = user_intro.cuda()
            gpt_description = gpt_description.cuda()

        loss, pred = trainer(texts, audios, videos, comments, c3d, user_intro, gpt_description)
        _, y = torch.max(pred, 1)
        diffusion_loss = loss.sum() / 1000.
        bce_loss_output = criterion(pred, labels)

        results.append(y)
        truths.append(labels)

        loss_output = (torch.abs(bce_loss_output - 0.1) + 0.1) + diffusion_loss*0.006
        total_loss += loss_output.item()
        optimizer.zero_grad()
        loss_output.backward(loss_output)
        optimizer.step()

    results = torch.cat(results)
    truths = torch.cat(truths)
    return total_loss, results, truths


def valid(loader, trainer, criterion, modelConfig: Dict):
    trainer.eval()
    results = []
    truths = []
    total_loss = 0.0
    total_batch_size = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            batch_size = batch["label"].size(0)
            texts = batch["text"]
            audios = batch["audioframes"]
            videos = batch["frames"]
            comments = batch["comments"]
            labels = batch["label"]
            c3d = batch["c3d"]
            user_intro = batch["user_intro"]
            gpt_description = batch["gpt_description"]
            total_batch_size += batch_size
            if torch.cuda.is_available():
                audios = audios.cuda()
                texts = texts.cuda()
                videos = videos.cuda()
                comments = comments.cuda()
                labels = labels.cuda()
                c3d = c3d.cuda()
                user_intro = user_intro.cuda()
                gpt_description = gpt_description.cuda()

            loss, pred = trainer(texts, audios, videos, comments, c3d, user_intro, gpt_description)
            _, y = torch.max(pred, 1)
            diffusion_loss = loss.sum() / 1000.
            bce_loss_output = criterion(pred, labels)
            
            results.append(y)
            truths.append(labels)

            loss_output = (torch.abs(bce_loss_output - 0.1) + 0.1) + diffusion_loss * 0.006
            total_loss += loss_output.item()

        results = torch.cat(results)
        truths = torch.cat(truths)
    return total_loss, results, truths