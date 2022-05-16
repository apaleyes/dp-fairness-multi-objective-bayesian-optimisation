import torch

from utils import get_device

def train(model, loss_funct, train_loader, optimizer, epoch, verbose = False, device = get_device(), log_file = None):
    model.train()

    train_loss = 0
    train_size = 0
    
    for inputs, target in train_loader:
        inputs, target = inputs.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(inputs.float())

        loss = loss_funct(output, target.float().reshape(-1, 1))

        train_loss += loss
        train_size += len(inputs)
        
        loss.backward()
        optimizer.step()

    if verbose:
      print(f'Epoch {epoch} | Loss {train_loss / train_size}')
        
def test(model, loss_funct, test_loader, device = get_device(), log_file = None, return_probs = False):
    model.eval()
    
    test_loss = 0
    correct = 0
    test_size = 0

    predictions = []
    prediction_prob = []
    
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)

            output = model(inputs.float())
            
            test_size += len(inputs)
            test_loss += loss_funct(output, target.float().reshape(-1, 1)).item()
            
            correct += torch.round(output).eq(target.view_as(output)).sum().item()

            predictions = predictions + torch.round(output).flatten().cpu().detach().tolist()
            prediction_prob = prediction_prob + output.flatten().cpu().detach().tolist()
            
    test_loss /= test_size
    accuracy = correct / test_size
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, test_size,
        100. * accuracy))
    
    if return_probs:
        return test_loss, accuracy, predictions, prediction_prob
    else:
        return test_loss, accuracy, predictions