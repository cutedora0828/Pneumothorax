#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import plot_confusion_matrix,confusion_matrix
import time
from tqdm import tqdm
from collections import defaultdict
from utils.Early_stopping import EarlyStopping
from utils.loss import FocalLoss
from utils.segmentationDataloader import DataLoaderSegmentation
from Model.Unet import Unet

#%%
torch.cuda.set_device(6)
#%%難的資料
train_dataset = DataLoaderSegmentation("/home/dora/data/pneumothorax_png/hard_train/")
test_dataset = DataLoaderSegmentation("/home/dora/data/pneumothorax_png/hard_test")
val_dataset = DataLoaderSegmentation("/home/dora/data/pneumothorax_png/hard_val/")

#%%
class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )

def train(train_loader, model, criterion, optimizer,scheduler, epoch, params):
    metric_monitor = MetricMonitor()
    model.train()
    loss1,train_loss = [], []
    stream = tqdm(train_loader)
    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(params["device"], non_blocking=True, dtype=torch.float)
        target = target.to(params["device"], non_blocking=True, dtype=torch.float)
        output = model(images)
        loss = criterion(output, target)
        loss1.append(loss.item())
        metric_monitor.update("Loss", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stream.set_description(
            "Epoch: {epoch}. Train. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
        )
    train_loss.append(np.array(loss1).mean())
    scheduler.step()
    return train_loss


def validate(val_loader, model, criterion, epoch, params, early_stopping):
    metric_monitor = MetricMonitor()
    model.eval()
    loss1, val_loss = [], []
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params["device"], non_blocking=True, dtype=torch.float)
            target = target.to(params["device"], non_blocking=True, dtype=torch.float)
            output = model(images)
            loss = criterion(output, target)
            loss1.append(loss.item())
            metric_monitor.update("Loss", loss.item())
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )
        val_loss.append(np.array(loss1).mean())
        early_stopping(np.array(loss1).mean(), model) #######這邊是np.array(loss1).mean()嗎？

    return early_stopping.early_stop, val_loss



def train_and_validate(model, train_dataset, val_dataset, params):
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        pin_memory=True,
        drop_last = True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        pin_memory=True,
        drop_last = True
    )
    criterion = params["criterion"].to(params["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1, last_epoch=-1)

    patience = params["patience"]
    early_stopping = EarlyStopping(patience, verbose=True)	
    train_loss = []
    val_loss = []
    for epoch in range(1, params["epochs"] + 1):
        train_loss.extend(train(train_loader, model, criterion, optimizer,scheduler, epoch, params))
        early_stopping.early_stop, val_loss = validate(val_loader, model, criterion, epoch, params, early_stopping)
        val_loss.extend(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    return model, train_loss, val_loss



#%%
start = time.time()
params = {
    "model": Unet(),
    "criterion" : FocalLoss(2),
    "top_score_threshold" : 0.3,
    "min_contour_area" : 300,
    "bottom_score_threshold" : 0.1,
    "device": "cuda",
    "lr": 0.0001,
    "batch_size": 64,
    "num_workers": 4,
    "epochs" : 50,
    "patience" : 45,
    "plot_img" : True,
    "number_plot" : None, 
    "safe_img" : True
}
model, train_loss, val_loss = train_and_validate(params["model"].to(params["device"]), train_dataset, val_dataset, params)
#%%
end = time.time()
t = (end - start)*0.000277777778
print("執行時間：%f 小時" % t)



#%%畫Loss圖的地方
safe_img = params["safe_img"]
plt.title("Unet Loss",fontsize = 20)
plt.plot(np.arange(len(train_loss)), train_loss, label='train')
plt.plot(np.arange(len(val_loss)), val_loss, label='Val')
plt.legend()
plt.ylabel("Value") 
plt.xlabel("Epoch")
if safe_img:
    plt.savefig('Unet_loss.png')
else:
    plt.show()
plt.close()

#%%
def IOU_function(predicted, mask):
    smooth = 1

    N = mask.shape[0]
    pred_flat = predicted.view(N, -1)
    gt_flat = mask.view(N, -1)
 
    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    IOU = (intersection + smooth) / ((unionset + smooth)-intersection)

    return IOU


#%%
#model = Unet().cuda()
#state_dict = torch.load('checkpoint.pt')
#model.load_state_dict(state_dict)

#%%
def predict(model, params, test_dataset, plot_img=False, number_plot = None, safe_img= False):
    test_loader = DataLoader(
        test_dataset, 
        batch_size=params["batch_size"], 
        shuffle=False, 
        num_workers=params["num_workers"], 
        pin_memory=True,
        drop_last = True
    )
    model.eval()
    predictions = []
    iou_accuracy = []
    outt = []
    mask = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(params["device"], dtype=torch.float)
            output = model(images)
            output = torch.sigmoid(output)

            predicted_masks = (output >= params["top_score_threshold"]).float()
            predicted_masks[predicted_masks.sum(axis=(1,2,3)) < params["min_contour_area"], ...] = torch.zeros_like(output[0])
            predicted_masks = (predicted_masks > params["bottom_score_threshold"] ).float()
            outt.extend(output.cpu().detach())
            #predicted_masks = torch.where(output>0.2,torch.ones_like(output),torch.zeros_like(output))

            predictions.extend(predicted_masks.cpu().detach())
            mask.extend(target.cpu())
            iou = IOU_function(predicted_masks.cpu(), target.cpu())
            iou_accuracy.extend(iou.numpy())
    

        if plot_img:
            if number_plot is not None:
                idx = number_plot
            else:
                idx = len(mask)
            
            #畫圖的地方
            for i in range(idx):
                plt.subplot(131)
                plt.imshow(mask[i].permute(1, 2, 0), cmap='gray')
                plt.title("Unet Mask")
                plt.subplot(132)
                plt.imshow((outt[i].permute(1,2,0)), cmap='gray')
                plt.title("output")
                plt.axis('off')
                plt.subplot(133)
                plt.imshow((predictions[i].permute(1,2,0)), cmap='gray')
                plt.title("Pred"+" IOU = "+str(round(iou_accuracy[i],3)))
                plt.axis('off')
                
                if safe_img:
                    plt.savefig("/home/dora/code/new_unet/prediction/unet_"+str(i+1)+"_IOU:"+str(round(iou_accuracy[i],2))+".png")
                else:
                    plt.show()
                plt.close()

    return iou_accuracy

iou_predictions = predict(params["model"].cuda(), params, test_dataset, plot_img=params["plot_img"], number_plot = params["number_plot"], safe_img=params["safe_img"])
print("IOU=",np.mean(iou_predictions))

#plt.imshow(predictions[10][1].permute(1,2,0))


# %%
# for i in range(10):
#     plt.imshow(train_dataset[i][0].permute(1,2,0), cmap='gray')
#     plt.show()

    