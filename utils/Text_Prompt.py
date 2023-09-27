# Code for "DeemCLIP: Dual Emotion Enhancement Module"



import torch
import clip
from transformers import pipeline

#classifier = pipeline("text-classification", model='/home/hhe2/projects/ActionCLIP/distilbert-base-uncased-emotion', return_all_scores=True)
classifier = pipeline("text-classification", model='j-hartmann/emotion-english-distilroberta-base', return_all_scores=True)
emo2id = {"anger":0,"disgust":1,"fear":2,"joy":3,"neutral":4,"sadness":5,"surprise":6}
# id2emo = {v:k for k, v in emo2id.items()}

def text_prompt(data):
    text_aug = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
                f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
                f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
                f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
                f"The man is {{}}", f"The woman is {{}}"]
    text_dict = {}
    text_dict_emo = {}
    num_text_aug = len(text_aug)

    for i, txt in enumerate(text_aug):
        text_dict[i] = [txt.format(c) for _, c in data.classes]

    for ii, txt in enumerate(text_aug):
        t = []
        for i, c in data.classes:
            t.append(txt.format(c))
        tt = classifier(t)
        t = []
        for i in tt:
            t.append(sorted(i, key=lambda x:x["score"])[-1]["label"])
        text_dict_emo[ii] = torch.cat([clip.tokenize(txt.format(c) + " " + t[i]) for i, c in data.classes])
    classes_emo = torch.cat([v for k, v in text_dict_emo.items()])
    return num_text_aug, text_dict, classes_emo, text_dict_emo


# def text_prompt_classes(text_dict_step1):
#     text_dict = {}
#     for i,txt_list in text_dict_step1.items():
#         text_dict[i] = torch.cat([clip.tokenize(txt) for txt in txt_list])
#     classes = torch.cat([v for k, v in text_dict.items()])
#     return classes


# def emo_label_logits(emo_text):
#     result = []
#     oupts = classifier(emo_text)
#     for oupt in oupts:
#         dist = [0]*6
#         for i in oupt:
#             dist[emo2id[i["label"]]] = i["score"]
#         result.append(dist)
#     log_softmax = torch.log(torch.tensor(result))
#     return log_softmax

def emo_label(emo_text):
    result = []
    oupts = classifier(emo_text)
    for i in oupts:
        result.append(emo2id[sorted(i,key=lambda x:x["score"])[-1]["label"]])
    return torch.tensor(result)