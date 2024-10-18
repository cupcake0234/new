from ..utils import *
from ..data_load.data_loader import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import tensor, from_numpy, no_grad, save, load, arange
from torch.autograd import Variable

class TrainBatchProcessor():
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg  # information of snapshot sequence
        '''prepare data'''
        if self.args.snapshot == 0:
            self.dataset = TrainDatasetMarginLoss(args, kg)
            self.data_loader = DataLoader(self.dataset,
                                      shuffle=True,
                                      batch_size=int(self.args.batch_size),
                                      collate_fn=self.dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.seed)),  # use seed generator
                                      pin_memory=True)
        elif self.args.snapshot > 0:
            self.rectify_dataset = TrainDatasetMarginLoss(args, kg, stage="rectify")
            self.rectify_data_loader = DataLoader(self.rectify_dataset,
                                      shuffle=True,
                                      batch_size=int(self.args.batch_size),
                                    #   batch_size=int(len(self.rectify_dataset)/ 4),
                                      collate_fn=self.rectify_dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.seed)),  # use seed generator
                                      pin_memory=True)

            self.retrain_dataset = TrainDatasetMarginLoss(args, kg, stage="retrain")
            self.retrain_data_loader = DataLoader(self.retrain_dataset,
                                      shuffle=True,
                                      batch_size=int(self.args.batch_size),
                                    #   batch_size=int(len(self.retrain_dataset)/ 4),
                                      collate_fn=self.retrain_dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.seed)),  # use seed generator
                                      pin_memory=True)
        
            self.remember_dataset = TrainDatasetMarginLoss(args, kg ,stage="memory")
            self.remember_data_loader = DataLoader(self.remember_dataset,
                                      shuffle=True,
                                      batch_size=int(self.args.batch_size),
                                    #   batch_size=int(len(self.remember_dataset)/ 4),
                                      collate_fn=self.remember_dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.seed)),  # use seed generator
                                      pin_memory=True)

    def process_epoch(self, model, optimizer):
        model.train()
        '''Start training'''
        total_loss = 0.0
        for idx_b, batch in enumerate(self.data_loader):
            '''get loss'''
            bh, br, bt, by = batch
            optimizer.zero_grad()
            batch_loss = model.loss(bh.to(self.args.device),
                                       br.to(self.args.device),
                                       bt.to(self.args.device),
                                       by.to(self.args.device) if by is not None else by).float()

            '''update'''
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            '''post processing'''
            model.epoch_post_processing(bh.size(0))
        return total_loss

    def process_rectify_epoch(self, model, optimizer):
        model.train()
        '''Start training'''
        total_loss, total_loss1, total_loss2 = 0.0, 0.0, 0.0
        batches = 0
        for idx_b, (batch, rectify_batch) in enumerate(zip(self.remember_data_loader, self.rectify_data_loader)):
            '''get loss'''
            bh, br, bt, by = batch
            rh, rr, rt, ry = rectify_batch
            optimizer.zero_grad()
            bh = bh.to(self.args.device)
            br = br.to(self.args.device)
            bt = bt.to(self.args.device)
            by = by.to(self.args.device)
            loss1 = model.loss(bh,
                                       br,
                                       bt,
                                       by if by is not None else by).float()
            loss2 = model.loss(rh.to(self.args.device),
                                       rr.to(self.args.device),
                                       rt.to(self.args.device),
                                       ry.to(self.args.device) if ry is not None else ry).float()
            loss = loss1 - self.args.beta1 * loss2

            '''update'''
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()

            batches += 1

        average_loss = total_loss/batches
        average_loss1 = total_loss1/batches
        average_loss2 = total_loss2/batches
        if self.args.epoch % 10 == 0:
            print(
                f"rectify epoch:\t{self.args.epoch} and loss:\t{average_loss}, loss1:{average_loss1}, loss2:{average_loss2}, beta1:{self.args.beta1}, samples:{len(self.remember_data_loader) + len(self.rectify_data_loader)}")
            '''post processing'''

            model.epoch_post_processing(bh.size(0))
        return total_loss

    def process_retrain_epoch(self, model, optimizer):
        model.train()
        '''Start training'''
        total_loss, total_loss1, total_loss2 = 0.0, 0.0, 0.0
        batches = 0
        for idx_b, (batch, retrain_batch) in enumerate(zip(self.remember_data_loader, self.retrain_data_loader)):
            '''get loss'''
            bh, br, bt, by = batch
            rh, rr, rt, ry = retrain_batch
            optimizer.zero_grad()
            loss1 = model.loss(bh.to(self.args.device),
                                       br.to(self.args.device),
                                       bt.to(self.args.device),
                                       by.to(self.args.device) if by is not None else by).float()
            loss2 = model.loss(rh.to(self.args.device),
                                       rr.to(self.args.device),
                                       rt.to(self.args.device),
                                       ry.to(self.args.device) if ry is not None else ry).float()
            loss = self.args.beta2 * loss1 + loss2

            '''update'''
            loss.backward()

            # 尝试冻住前snapshot的关系向量
            # if self.args.Plan_weight == "True" :
            #     # 冻住上一个snapshot的关系向量（需要改进为稳定关系？？？）
            #     mask = torch.ones_like(model.rel_embeddings.grad)
            #     mask[:self.kg.snapshots[self.args.snapshot - 1].num_rel] = 0
            #     model.rel_embeddings.grad = model.rel_embeddings.grad * mask
                
            optimizer.step()

            total_loss += loss.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()

            batches += 1

        average_loss = total_loss / batches
        average_loss1 = total_loss1 / batches
        average_loss2 = total_loss2 / batches
        if self.args.epoch % 10 == 0:
            print(
                f"retrain epoch:\t{self.args.epoch} and loss:\t{average_loss}, loss1:{average_loss1}, loss2:{average_loss2}, beta1:{self.args.beta1}, samples:{len(self.remember_data_loader) + len(self.retrain_data_loader)}")
            '''post processing'''

            model.epoch_post_processing(bh.size(0))
        return total_loss

class DevBatchProcessor():
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg  # information of snapshot sequence
        self.batch_size = 100
        '''prepare data'''
        self.dataset = TestDataset(args, kg)
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=False,
                                      batch_size=self.batch_size,
                                      collate_fn=self.dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.seed)),
                                      pin_memory=True)

    def process_epoch(self, model):
        model.eval()
        num = 0
        results = dict()
        sr2o = self.kg.snapshots[self.args.snapshot].sr2o_all
        '''start evaluation'''
        for step, batch in enumerate(self.data_loader):
            sub, rel, obj, label = batch
            sub = sub.to(self.args.device)
            rel = rel.to(self.args.device)
            obj = obj.to(self.args.device)
            label = label.to(self.args.device)
            num += len(sub)
            if self.args.valid:
                stage = 'Valid'
            else:
                stage = 'Test'
            '''link prediction'''
            pred = model.predict(sub, rel, stage=stage)

            b_range = torch.arange(pred.size()[0], device=self.args.device)
            target_pred = pred[b_range, obj]
            pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)

            pred[b_range, obj] = target_pred

            '''rank all candidate entities'''
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                b_range, obj]
            '''get results'''
            ranks = ranks.float()
            results['count'] = torch.numel(ranks) + results.get('count', 0.0)
            results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
            results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
            for k in range(10):
                results['hits{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                    'hits{}'.format(k + 1), 0.0)
        count = float(results['count'])
        for key, val in results.items():
            results[key] = round(val / count, 4)
        return results


class DevBatchProcessor_MEANandLAN():
    '''
    To save memory, we collect the queries with the same relation and then perform evaluation.
    '''
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg  # information of snapshot sequence
        self.batch_size = 1
        '''prepare data'''
        self.dataset = TestDataset(args, kg)
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=False,
                                      batch_size=self.batch_size,
                                      collate_fn=self.dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.seed)),
                                      pin_memory=True)

    def process_epoch(self, model):
        model.eval()
        num = 0
        results = dict()
        '''start evaluation'''
        sub, rel, obj, label = None, None, None, None

        for step, batch in enumerate(self.data_loader):
            sub_, rel_, obj_, label_ = batch
            if sub == None:
                sub, rel, obj, label = sub_, rel_, obj_, label_
                continue
            elif rel[0] == rel_ and rel.size(0) <= 50:
                sub = torch.cat((sub, sub_), dim=0)
                rel = torch.cat((rel, rel_), dim=0)
                obj = torch.cat((obj, obj_), dim=0)
                label = torch.cat((label, label_), dim=0)
                continue

            sub = sub.to(self.args.device)
            rel = rel.to(self.args.device)
            obj = obj.to(self.args.device)
            label = label.to(self.args.device)
            num += len(sub)
            if self.args.valid:
                stage = 'Valid'
            else:
                stage = 'Test'
            '''link prediction'''
            pred = model.predict(sub, rel, stage=stage)

            b_range = torch.arange(pred.size()[0], device=self.args.device)
            target_pred = pred[b_range, obj]
            pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)

            pred[b_range, obj] = target_pred

            '''rank all candidate entities'''
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                b_range, obj]
            '''get results'''
            ranks = ranks.float()
            results['count'] = torch.numel(ranks) + results.get('count', 0.0)
            results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
            results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
            for k in range(10):
                results['hits{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                    'hits{}'.format(k + 1), 0.0)
            sub, rel, obj, label = None, None, None, None

        count = float(results['count'])
        for key, val in results.items():
            results[key] = round(val / count, 4)
        return results