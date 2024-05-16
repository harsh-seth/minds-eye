import argparse
import torch

from transformers import set_seed
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data.Pix3dDataset import Pix3dDataset
from model.MindsEye import MindsEye
from utils.data_utils import make_dir_path, save_checkpoint

def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description="CLI for training Mind's Eye")

    parser.add_argument('--evaluate', '-e', action="store_true",
                        help='Only run evaluation')
    
    parser.add_argument('--triangles', action="store_true",
                        help='Pass to run scripts for Triangles instead of Vertices')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='mini-batch size (default: 16)')

    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs (default: 100)')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (AdamW) (default: 1e-4)')

    parser.add_argument('--wd', type=float, default=0.001,
                        help='Weight Decay (AdamW) (default: 0.001)')

    parser.add_argument('--device', default='cuda', type=str,
                        help='device to be used for computations (in {cpu, cuda:0, cuda:1, ...}, default: cuda)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for random initialization (default: 42)')
    
    parser.add_argument('--from_checkpoint', type=int, default=None,
                        help='Resume from checkpoint @ specified epoch number')

    parser.add_argument('--data_dir', type=str, default="./data/pix3d", 
                        help="Folder where the pix3d dataset and generated grid images are stored")

    parser.add_argument('--train_split', type=float, default=0.85,
                        help="Determines train/test split. Between [0.0, 1.0], default: 0.85")
    
    parser.add_argument('--max_records', type=int, default=None,
                        help="Determines the max number of records to a build dataset on, default: all records")

    parser.add_argument('--max_vertices', type=int, default=4000,
                    help="Determines the max number of vertices to allow in a single mesh, default: 4000")

    parser.add_argument('--max_triangles', type=int, default=750,
                    help="Determines the max number of triangles to allow in a single mesh, default: 750")
    
    parsed_arguments = parser.parse_args()
    return parsed_arguments

def evaluateOnce(model, criterion, dataloader, device='cuda', generate_triangles=False):
    model.to(device)
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for batch in tqdm(dataloader):
            inputs = batch[0]
            targets = batch[1][int(generate_triangles)]
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss/len(dataloader)

def trainOnce(model, criterion, optimizer, dataloader, device='cuda', generate_triangles=False):
    model.to(device)
    model.train()
    epoch_train_loss = 0.0
    for batch in tqdm(dataloader):
        optimizer.zero_grad()

        inputs = batch[0]
        targets = batch[1][int(generate_triangles)]
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        epoch_train_loss += loss.item()

        loss.backward()
        optimizer.step()
    return epoch_train_loss/len(dataloader)

def train(model, criterion, optimizer, train_dataloader, validation_dataloader, num_epochs, device='cuda', starting_epoch=0, save_path_prefix="results", generate_triangles=False):
    for epoch in range(starting_epoch, num_epochs):
        print(f"[Epoch {epoch + 1} / {num_epochs}]")

        train_loss = trainOnce(model, criterion, optimizer, train_dataloader, device, generate_triangles=generate_triangles)
        print(f"\t Train Loss = {train_loss:.4f}")

        if (epoch+1) % 10 == 0:
            save_checkpoint(model, f"{save_path_prefix}/checkpoints/checkpoint-{epoch+1}.pth")
        
        with open(f"{save_path_prefix}/metrics.csv", "a") as results_file:
            results_file.write(f"train,{epoch+1},{train_loss:.4f}\n")
        
        val_loss = evaluateOnce(model, criterion, validation_dataloader, device, generate_triangles=generate_triangles)
        print(f"\t Val Loss = {val_loss:.4f}")
        with open(f"{save_path_prefix}/metrics.csv", "a") as results_file:
            results_file.write(f"val,{epoch+1},{val_loss:.4f}\n")
    
    save_checkpoint(model, f"{save_path_prefix}/checkpoints/checkpoint-{epoch+1}.pth")

if __name__ == '__main__':
    args = parse_command_line_arguments()
    for k, v in args.__dict__.items():
        print(k + '=' + str(v))
    
    set_seed(args.seed)
    save_path_prefix = "results/triangles" if args.triangles else "results/vertices"
    make_dir_path(save_path_prefix)
    
    dataset = Pix3dDataset(data_dir=args.data_dir, max_records=args.max_records, max_vertices=args.max_vertices, max_triangles=args.max_triangles)
    train_dataset, validation_dataset = random_split(dataset, [args.train_split, 1.0 - args.train_split])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Dataloaders ready. Train set length: {len(train_dataloader)}, Validation set length: {len(validation_dataloader)}")

    model = torch.load(f'{save_path_prefix}/checkpoints/checkpoint-{args.from_checkpoint}.pth', map_location=lambda storage, location: storage) if args.from_checkpoint else MindsEye(generate_triangles=args.triangles, num_vertices=args.max_vertices, num_triangles=args.max_triangles)
    criterion = nn.L1Loss()
    if args.evaluate:
        test_loss = evaluateOnce(model, criterion, validation_dataloader, args.device, generate_triangles=args.triangles)
        print(f"\t Test Loss = {test_loss:.4f}")
        with open(f"{save_path_prefix}/metrics.csv", "a") as results_file:
            results_file.write(f"test,-1,{test_loss:.4f}\n")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        train(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                validation_dataloader=validation_dataloader,
                num_epochs=args.epochs,
                device=args.device,
                starting_epoch=args.from_checkpoint if args.from_checkpoint else 0,
                save_path_prefix=save_path_prefix,
                generate_triangles=args.triangles,
            )
