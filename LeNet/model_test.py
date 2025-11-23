import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet


def test_data_process():
    test_data = FashionMNIST('./data',
                              train=False,
                              download=True,
                              transform=transforms.Compose([transforms.Resize(size=28),
                                                            transforms.ToTensor()]))

    test_loader = Data.DataLoader(test_data,
                                   batch_size=1,
                                   shuffle=True,
                                   num_workers=0)
    return test_loader

def test_model_process(model,test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    correct = 0
    total = 0
    #不计算梯度 只4进行前向传播
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            model.eval()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.sum(predicted == labels)

    test_accuracy = 100.0 * correct.double().item() / total
    print("Test Accuracy of the model on the 10000 test images: {} %".format(test_accuracy))


if __name__ == "__main__":
    LeNet = LeNet()
    LeNet.load_state_dict(torch.load('./model.pth'))
    test_loader = test_data_process()
    # test_model_process(LeNet,test_loader)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LeNet.to(device)

    classes = ['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            model.eval()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            result = predicted.item()
            label = labels.item()

            print("预测值：",classes[result],"-------","真实值：",classes[label])




