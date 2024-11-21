from dataset import *

if __name__ == "__main__":

    data_path = "/Users/fionachow/Documents/NYU/CDS/Fall 2024/DS - GA 1008 Deep Learning/Project/DL_Final_Proj/data/subset/"  

    train_loader = create_wall_encoder_dataloader(data_path, split="train", batch_size=64)
    val_loader = create_wall_encoder_dataloader(data_path, split="val", batch_size=64)
    test_loader = create_wall_encoder_dataloader(data_path, split="test", batch_size=64)

    print("Train loader size:", len(train_loader))
    print("Validation loader size:", len(val_loader))
    print("Test loader size:", len(test_loader))
