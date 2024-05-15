import argparse
import tqdm
import torch
from torch import nn

from transformers import set_seed

from model.ImageGridEncoder import ImageGridEncoder
from utils.data_utils import save_checkpoint

def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description="CLI for training Mind's Eye")

    parser.add_argument('--evaluate', '-e', action="store_true",
                        help='Only run evaluation')

    parser.add_argument('--batch_size', type=int, default=2,
                        help='mini-batch size (default: 2)')

    parser.add_argument('--epochs', type=int, default=2,
                        help='number of training epochs (default: 2)')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (AdamW) (default: 1e-4)')

    parser.add_argument('--wd', type=float, default=0.001,
                        help='Weight Decay (AdamW) (default: 0.001)')

    parser.add_argument('--workers', type=int, default=0,
                        help='number of working units used to load the data (default: 0)')

    parser.add_argument('--device', default='cuda', type=str,
                        help='device to be used for computations (in {cpu, cuda:0, cuda:1, ...}, default: cuda)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for random initialization (default: 42)')
    
    parser.add_argument('--resume_from_epoch', type=int, default=None,
                help='Resume from checkpoint @ specified epoch number')
    
    parsed_arguments = parser.parse_args()
    return parsed_arguments

def evaluate(model, criterion, dataloader, device='cuda'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for batch in tqdm(dataloader):
            inputs = batch.input_vec
            targets = batch.output_vec
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss/len(dataloader)

def train(model, criterion, optimizer, train_dataloader, validation_dataloader, num_epochs, device='cuda', starting_epoch=0, save_path_prefix="results"):
    model.to(device)
    for epoch in range(starting_epoch, num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()

            inputs = batch.input_vec
            targets = batch.output_vec
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            epoch_train_loss += loss.item()

            loss.backward()
            optimizer.step()
        print(f"[Epoch {epoch + 1} / {num_epochs}]")
        train_loss = epoch_train_loss/len(train_dataloader)
        print(f"\t Train Loss = {train_loss:.4f}")

        if (epoch+1) % 10 == 0:
            save_checkpoint(model, f"{save_path_prefix}/checkpoints/checkpoint-{epoch+1}.pth")
        
        with open(f"{save_path_prefix}/metrics.csv", "a") as results_file:
            results_file.write(f"train,{epoch+1},{train_loss:.4f}\n")
        
        model.eval()
        val_loss = evaluate(model, criterion, validation_dataloader, device)
        print(f"\t Val Loss = {val_loss:.4f}")
        with open(f"{save_path_prefix}/metrics.csv", "a") as results_file:
            results_file.write(f"val,{epoch+1},{val_loss:.4f}\n")
    
    torch.save(model, f"{save_path_prefix}/checkpoints/checkpoint-{epoch+1}.pth")
    save_checkpoint(model, f"{save_path_prefix}/checkpoints/checkpoint-{epoch+1}.pth")

if __name__ == '__main__':
    args = parse_command_line_arguments()
    for k, v in args.__dict__.items():
        print(k + '=' + str(v))
    
    set_seed(args.seed)
    save_path_prefix = "results"
    
    model = torch.load(f'{save_path_prefix}/checkpoints/checkpoint-{args.resume_from_epoch}.pth', map_location=lambda storage, location: storage) if args.resume_from_epoch else ImageGridEncoder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.L1Loss()

    # TODO: Solve dataloading
    dataset = None
    train_set = dataset.get('train')
    validation_set = dataset.get('validation')
    test_set = dataset.get('test')

    # TODO: Solve dataloader
    train_dataloader = train_set
    validation_dataloader = validation_set
    test_dataloader = test_set

    if not args.evaluate:
        train(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                train_set=train_set,
                validation_set=validation_set,
                num_epochs=args.epochs,
                device=args.device,
                batch_size=args.batch_size,
                starting_epoch=args.resume_from_epoch if args.resume_from_epoch else 0,
                save_path_prefix=save_path_prefix,
            )

    test_loss = evaluate(model, criterion, test_dataloader, args.device)
    print(f"\t Test Loss = {test_loss:.4f}")
    with open(f"{save_path_prefix}/metrics.csv", "a") as results_file:
        results_file.write(f"test,-1,{test_loss:.4f}\n")
