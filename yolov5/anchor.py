from utils.general import check_dataset
from utils.autoanchor import kmean_anchors
from utils.dataloaders import create_dataloader


if __name__ == "__main__":
    data = 'script/dataset.yaml'
    train_path = check_dataset(data)['train']
    _, dataset = create_dataloader(train_path, 640, 8, 32, workers=0, shuffle=True)
    anchors = kmean_anchors(dataset, n=9, img_size=640, thr=4.0, gen=1000, verbose=False)
