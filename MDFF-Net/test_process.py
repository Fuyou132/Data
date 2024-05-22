import torch

def test_process(model, device, test_loader, epoch, celoss):
    model.eval()
    test_loss = 0
    correct = 0

    with open('./results/test_results/test_results_'+str(epoch).zfill(3)+'.txt', 'w') as f:
        f.write('')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            test_loss += celoss(torch.unsqueeze(output, 0), target.long()).item()
            pred = torch.unsqueeze(output, 0).max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            with open('./results/test_results/test_results_'+str(epoch).zfill(3)+'.txt', 'a+') as f:
                f.write(str(target.to('cpu').item()) + ', ' + str(pred.to('cpu').item())+'\t\n')

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            accuracy))
    return accuracy if accuracy else 0.0

