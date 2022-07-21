from dataset import *
from train import *
from UNET import *


def main():

    unet = UNET(4, 2)

    base_path = Path("../input/38-Cloud_training")
    dataset = CloudDastaset(
        base_path / "train_red",
        base_path / "train_green",
        base_path / "train_blue",
        base_path / "train_nir",
        base_path / "train_gt",
    )
    # print("dataset has a length of: ", len(dataset))
    # print('torch: ',torch.__version__)
    # print('torchcuda available: ',torch.cuda.is_available())
    # print(torch.__file__)

    train_val_size = (int(len(dataset) * 7 / 10)), int(len(dataset) * 3 / 10)
    train_ds, valid_ds = torch.utils.data.random_split(dataset, train_val_size)
    train_dl = DataLoader(train_ds, batch_size=10, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=10, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(unet.parameters(), lr=0.01)

    
    train_loss, valid_loss = train(
        unet, train_dl, valid_dl, loss_fn, opt, acc_metric, epochs=25
    )
    


if __name__ == "__main__":
    main()
