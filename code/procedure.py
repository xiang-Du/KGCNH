import torch
from utils import calc_auc, calc_aupr


def train(model, kg_graph, augment_graph, kg_relation, augment_relation, train_dataloader, optimizer, device):
    model.train()
    avg_loss = 0
    avg_reg_loss = 0
    size = len(train_dataloader)
    for i, data in enumerate(train_dataloader):
        h, t, neg_t, _, _ = data
        h = h.to(device)
        t = t.to(device)
        neg_t = neg_t.to(device)
        loss, reg_loss = model.train_step(kg_graph, augment_graph, kg_relation, augment_relation, h, t,
                                          neg_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.detach().to('cpu').item()
        avg_reg_loss += reg_loss.detach().to('cpu').item()
    return avg_loss / size, avg_reg_loss / size


def test(model, kg_graph, graph, relation, g_relation, valid_dataloader, device):
    model.eval()
    all_logit = []
    all_label = []
    with torch.no_grad():
        for i, data in enumerate(valid_dataloader):
            h, t, neg_t, pos_label, neg_label = data
            h = h.to(device)
            t = t.to(device)
            neg_t = neg_t.to(device)
            score, embed = model.predict(kg_graph, graph, relation, g_relation, h, t, neg_t)
            all_logit = all_logit + score.to('cpu').tolist()
            label = torch.cat((pos_label.view(-1).to('cpu'), neg_label.view(-1).to('cpu')))
            all_label = all_label + label.tolist()
    auc = calc_auc(all_label, all_logit)
    aupr = calc_aupr(all_label, all_logit)
    return auc, aupr, all_logit, all_label, embed
