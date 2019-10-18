from model_s import *
import matplotlib.pyplot as plt

num_epochs = 1001
batch_size = 1024
learning_rate = 1e-3
transform_train = transforms.Compose([
        ReadNP(0),
        RandomCrop2D(8),
        RandomLogNorm(0),
        RandomAddNoise(0.),
        transforms.ToTensor()
    ])

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda tensor:min_max_normalization(tensor, 0, 1)),
])

df_training = torchvision.datasets.DatasetFolder('../eor_data/',
                                        loader = np.load, extensions='npz',
                                                 transform=transform_train)
dataloader = torch.utils.data.DataLoader(df_training,
                                           batch_size=batch_size, shuffle=True)


model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

for epoch in range(num_epochs+1):
    for data in dataloader:
        img, _ = data
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        print(img.shape)
        print(output.shape)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.item()))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './dc_img/image.png')#.format(epoch))
        pic = to_img(img.cpu().data)
        save_image(pic, './dc_img/image_1.png')#.format(epoch))
        torch.save(model, './conv_autoencoder3.pth')
    if epoch % 100 == 0:
        x = model.latent(img)
        x = x.cpu().detach().numpy()
        # print(x)
        print(x.shape)
        np.savez('test.npz', x=x)
