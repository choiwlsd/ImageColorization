import os #os임포트
import random #random임포트
import torch 
import torchvision
import torch.nn as nn
from torch.backends import cudnn
from torch import optim
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
from torchvision.datasets.folder import default_loader
import torch.nn.functional as F #~pytorch임포트
import numpy as np #numpy임포트
from PIL import Image #PIL에서 image임포트
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg #~matplotlib임포트
from skimage.color import lab2rgb, rgb2lab, rgb2gray #skimage임포트
from skimage import io

from tqdm import tqdm

use_gpu = torch.cuda.is_available() #GPU사용 유무

class Generator(nn.Module): #생성 모델 구성
    def __init__(self, batch_size): #생성모델 구성
        super(Generator, self).__init__()

        # [1x256x256] -> [64x128x128]
        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1) #흑백 이미지 인풋 -> [64x128x128]
        
        conv2 = [nn.LeakyReLU(0.2, inplace=True), #leakyReLU통과
            nn.Conv2d(64, 128, 4, 2, 1)] #[64x128x128] -> [128x64x64]

        conv2 += [nn.BatchNorm2d(128)] #batchnorm 통과

        self.conv2 = nn.Sequential(*conv2) #이어붙이기

        
        conv3 = [nn.LeakyReLU(0.2, inplace=True),#leakyReLU통과
                 nn.Conv2d(128, 256, 4, 2, 1)]# -> [256x32x32]

        conv3 += [nn.BatchNorm2d(256)]#batchnorm 통과

        self.conv3 = nn.Sequential(*conv3)#이어붙이기

        
        conv4 = [nn.LeakyReLU(0.2, inplace=True),#leakyReLU통과
                 nn.Conv2d(256, 512, 4, 2, 1)]# -> [512x16x16]
        conv4 += [nn.BatchNorm2d(512)]#batchnorm 통과
        self.conv4 = nn.Sequential(*conv4)#이어붙이기

        
        conv5 = [nn.LeakyReLU(0.2, inplace=True),#leakyReLU통과
                 nn.Conv2d(512, 512, 4, 2, 1)]# -> [512x8x8]
        conv5 += [nn.BatchNorm2d(512)]#batchnorm 통과
        self.conv5 = nn.Sequential(*conv5)#이어붙이기

        
        conv6 = [nn.LeakyReLU(0.2, inplace=True),#leakyReLU통과
                 nn.Conv2d(512, 512, 4, 2, 1)]# -> [512x4x4]
        conv6 += [nn.BatchNorm2d(512)]#batchnorm 통과
        self.conv6 = nn.Sequential(*conv6)#이어붙이기

        
        conv7 = [nn.LeakyReLU(0.2, inplace=True),#leakyReLU통과
                 nn.Conv2d(512, 512, 4, 2, 1)]# -> [512x2x2]
        conv7 += [nn.BatchNorm2d(512)]#batchnorm 통과
        self.conv7 = nn.Sequential(*conv7)#이어붙이기

        
        conv8 = [nn.LeakyReLU(0.2, inplace=True),#leakyReLU통과
                 nn.Conv2d(512, 512, 4, 2, 1)]# -> [512x1x1]
        conv8 += [nn.BatchNorm2d(512)]#batchnorm 통과
        self.conv8 = nn.Sequential(*conv8)#이어붙이기

        
        deconv8 = [nn.ReLU(),#ReLU통과
                   nn.ConvTranspose2d(512, 512, 4, 2, 1)]#디컨볼루션 -> [512x2x2]
        deconv8 += [nn.BatchNorm2d(512), nn.Dropout(0.5)]#batchnorm 통과+dropout
        self.deconv8 = nn.Sequential(*deconv8)#이어붙이기

        
        deconv7 = [nn.ReLU(),#ReLU통과
                   nn.ConvTranspose2d(512 * 2, 512, 4, 2, 1)]#디컨볼루션 [(512+512)x2x2] -> [512x4x4]
        deconv7 += [nn.BatchNorm2d(512), nn.Dropout(0.5)]#batchnorm 통과+dropout
        self.deconv7 = nn.Sequential(*deconv7)#이어붙이기

        
        deconv6 = [nn.ReLU(),#ReLU통과
                   nn.ConvTranspose2d(512 * 2, 512, 4, 2, 1)]#디컨볼루션 [(512+512)x4x4] -> [512x8x8]
        deconv6 += [nn.BatchNorm2d(512), nn.Dropout(0.5)]#batchnorm 통과+dropout
        self.deconv6 = nn.Sequential(*deconv6)#이어붙이기

        
        deconv5 = [nn.ReLU(),#ReLU통과
                   nn.ConvTranspose2d(512 * 2, 512, 4, 2, 1)]#디컨볼루션 [(512+512)x8x8] -> [512x16x16]
        deconv5 += [nn.BatchNorm2d(512)]#batchnorm 통과
        self.deconv5 = nn.Sequential(*deconv5)#이어붙이기

        
        deconv4 = [nn.ReLU(),#ReLU통과
                   nn.ConvTranspose2d(512 * 2, 256, 4, 2, 1)]#디컨볼루션 [(512+512)x16x16] -> [256x32x32]
        deconv4 += [nn.BatchNorm2d(256)]#batchnorm 통과
        self.deconv4 = nn.Sequential(*deconv4)#이어붙이기

        
        deconv3 = [nn.ReLU(),#ReLU통과
                   nn.ConvTranspose2d(256 * 2, 128, 4, 2, 1)]#디컨볼루션 [(256+256)x32x32] -> [128x64x64]
        deconv3 += [nn.BatchNorm2d(128)]#batchnorm 통과
        self.deconv3 = nn.Sequential(*deconv3)#이어붙이기

        
        deconv2 = [nn.ReLU(),#ReLU통과
                   nn.ConvTranspose2d(128 * 2, 64, 4, 2, 1)]#디컨볼루션 [(128+128)x64x64] -> [64x128x128]
        deconv2 += [nn.BatchNorm2d(64)]#batchnorm 통과
        self.deconv2 = nn.Sequential(*deconv2)#이어붙이기

        
        self.deconv1 = nn.Sequential(
            nn.ReLU(),#ReLU통과
            nn.ConvTranspose2d(64 * 2, 2, 4, 2, 1),#디컨볼루션 [(64+64)x128x128] -> [3x256x256]
            nn.Tanh()#tanh통과
        )


    def forward(self, x):

        c1 = self.conv1(x) 
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)
        c7 = self.conv7(c6)
        c8 = self.conv8(c7)#~각 층별로 통과시키기

        d7 = self.deconv8(c8)
        d7 = torch.cat((c7, d7), dim=1) 
        d6 = self.deconv7(d7)
        d6 = torch.cat((c6, d6), dim=1)
        d5 = self.deconv6(d6)
        d5 = torch.cat((c5, d5), dim=1)
        d4 = self.deconv5(d5)
        d4 = torch.cat((c4, d4), dim=1)
        d3 = self.deconv4(d4)
        d3 = torch.cat((c3, d3), dim=1)
        d2 = self.deconv3(d3)
        d2 = torch.cat((c2, d2), dim=1)
        d1 = self.deconv2(d2)
        d1 = torch.cat((c1, d1), dim=1) #~u-net구조, 같은 레벨의 컨볼루션층과 합.
        out = self.deconv1(d1)#합한것을 dconv통과

        return out

class Discriminator(nn.Module): #분별 모델 구성
    def __init__(self, batch_size):
        super(Discriminator, self).__init__()
        # [(3+3)x256x256] -> [64x128x128] -> [128x64x64]
        main = [nn.Conv2d(3, 64, 4, 2, 1), #3채널 입력
            nn.LeakyReLU(0.2, inplace=True), #LeakyReLU 통과
            nn.Conv2d(64, 128, 4, 2, 1)] #컨볼루션 통과
        main += [nn.BatchNorm2d(128)] #batchnorm통과

        # -> [256x32x32]
        main += [nn.LeakyReLU(0.2, inplace=True),#LeakyReLU 통과
                  nn.Conv2d(128, 256, 4, 2, 1)]#컨볼루션 통과
        main += [nn.BatchNorm2d(256)]#batchnorm통과

        # -> [512x31x31] (Fully Convolutional)
        main += [nn.LeakyReLU(0.2, inplace=True),#LeakyReLU 통과
                  nn.Conv2d(256, 512, 4, 1, 1)]#컨볼루션 통과
        main += [nn.BatchNorm2d(512)]#batchnorm통과

        # -> [1x30x30] (Fully Convolutional, PatchGAN)
        main += [nn.LeakyReLU(0.2, inplace=True),#LeakyReLU 통과
                  nn.Conv2d(512, 1, 4, 1, 1),#컨볼루션 통과
                  nn.Sigmoid()] #sigmoid 통과

        self.main = nn.Sequential(*main)

    def forward(self, x1, x2): 
        out = torch.cat((x1, x2), dim=1) #가짜 하나, 진짜 하나 합쳐서 통과
        return self.main(out)


global args_tmp #전역으로 사용되는 변수 지정
args_tmp={}
args_tmp['dataroot'] = 'data' #데이터셋 경로


args_tmp['num_epochs'] = 100 #epoch수
args_tmp['batchSize'] = 8 #batch크기
args_tmp['lr'] = 0.0002 #학습률
args_tmp['beta1'] = 0.5
args_tmp['beta2'] = 0.999 #~adam에 사용되는 모멘텀 변수
args_tmp['lambda_A'] = 100.0 #L1항 계수

args_tmp['model_path'] = './models' #모델 경로
args_tmp['sample_path'] = './results' #샘플 경로
args_tmp['log_step'] = 1 #10번마다 log찍기
args_tmp['sample_step'] = 100  #100번마다 샘플 뽑기
args_tmp['num_workers'] = 2 #dataloader worker수

args_tmp['save_freq'] = 20

class ImageFolder_lab(datasets.ImageFolder): #lab으로 뽑아주는 imagefolder클래스
    def __init__(self, root, transform):
        # os.listdir Function gives all lists of directory
        self.root = root #root = root
        self.transform = transform #받은 transform 그대로 사용
        self.loader=default_loader #default_loader 그대로 사용
        self.dir_AB = os.path.join(root)  # 학습 데이터셋 경로 지정
        self.image_paths = list(map(lambda x: os.path.join(self.dir_AB, x), os.listdir(self.dir_AB))) #학습 파일 경로 지정
        

    def __getitem__(self, index):#getitem 오버로딩
        AB_path = self.image_paths[index] #이미지 파일 경로 지정
        img = self.loader(AB_path) #이미지 로드
       

        img_original = self.transform(img) #transform 통과
        img_original = img_original.numpy()
        img_original = np.asarray(img_original) #nparray로 변환
        lab = torch.from_numpy(img_original.transpose((1, 2, 0))).float()
        target = torch.from_numpy(img_original.transpose((2, 0, 1))).float() #채널수를 뒤로 뺌
        img_lab = rgb2lab(lab) #lab으로 변환
        img_lab = (img_lab + 128) / 255 #정규화
        img_ab = img_lab[:, :, 1:3] #ab채널만 뽑기
        img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float() #채널수 다시 뽑기
        img_original = rgb2gray(lab) #흑백이미지 생성
        img_original = torch.from_numpy(img_original).unsqueeze(0).float() #흑백이미지 torch변수로 변환
        return img_original, img_ab, target #리턴

    def __len__(self):
        return len(self.image_paths) #dataset길이 리턴

def to_variable(x): #variable로 변환하는 helper 함수
    if torch.cuda.is_available(): #GPU있으면 
        x = x.cuda() #cuda변수로 변환
    return Variable(x) # 리턴

def denorm(x): #비정규화 helper 함수
    out = (x + 1) / 2 
    return out.clamp(0, 1) #정규화 과정 반대로 수행

def GAN_Loss(input, target, criterion): #GAN 손실 연산 함수
    if target == True: #진짜 이미지이면 
        tmp_tensor = torch.FloatTensor(input.size()).fill_(1.0)
        labels = Variable(tmp_tensor, requires_grad=False) #1로 가득 찬 variable 생성
    else:#가짜 이미지이면
        tmp_tensor = torch.FloatTensor(input.size()).fill_(0.0)
        labels = Variable(tmp_tensor, requires_grad=False) #0으로 가득 찬 variable 생성

    if torch.cuda.is_available(): #GPU사용가능하면 변환
        labels = labels.cuda()

    return criterion(input, labels) #Loss함수로 넘겨주기
 
def to_rgb(grayscale_input, ab_input, real_ab , save_path=None, save_name=None): #수행중 rgb로 변환, 출력
    plt.clf() # clear matplotlib
    plt.subplots(nrows=4, ncols=3) #격자 구조 생성
    for i in range(4): #4번 반복
        gray_temp = grayscale_input[i] #흑백 이미지 임시 저장
        color_image = torch.cat((grayscale_input[i], ab_input[i]), 0).numpy() # 흑백, ab채널 합치기
        color_image = color_image.transpose((1, 2, 0))  # 채널 뒤로 빼기
        color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100 
        color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   #~정규화 풀고 RGB범위 맞춰주기
        color_image = lab2rgb(color_image.astype(np.float64)) #lab에서 rgb로
        real_image = torch.cat((grayscale_input[i], real_ab[i]), 0).numpy() # 흑백, ab채널 합치기
        real_image = real_image.transpose((1, 2, 0))  # 채널 뒤로 빼기
        real_image[:, :, 0:1] = real_image[:, :, 0:1] * 100
        real_image[:, :, 1:3] = real_image[:, :, 1:3] * 255 - 128   #~정규화 풀고 RGB범위 맞춰주기
        real_image = lab2rgb(real_image.astype(np.float64)) #lab에서 rgb로
        gray_temp = gray_temp.squeeze().numpy()
        plt.subplot(4,3,i*3+1) #흑백 출력
        #print("gray : ", grayscale_input.shape)
        plt.imshow(gray_temp, cmap='gray')#흑백 출력
        plt.grid(None) #격자 없애기
        plt.subplot(4,3,i*3+2) #생성 이미지 출력
        plt.imshow(color_image)#생성 이미지 출력
        plt.grid(None)#격자 없애기
        plt.subplot(4,3,i*3+3)#자연 이미지 출력
        plt.imshow(real_image)#자연 이미지 출력
        plt.grid(None)#격자 없애기
    plt.show()
    if save_path is not None and save_name is not None:  #각 이미지 출력~
        plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
        plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))
        plt.imsave(arr=real_image, fname='{}{}'.format(save_path['real'], save_name))

def main(): #main함수

    cudnn.benchmark = True
    #print(args_tmp)
    transform_t = transforms.Compose([transforms.ToTensor()]) #transform 선언, (imageset클래스로 옮김)

    dataset = ImageFolder_lab('data', transform_t) #dataset 선언.
    data_loader = data.DataLoader(dataset=dataset, #loader 선언, dataset은 위에서 선언한것 사용
                                  batch_size=args_tmp['batchSize'], #batch_size는 위에서 선언한것 사용
                                  shuffle=True, #섞기
                                  num_workers=args_tmp['num_workers']) #num_worker만큼 사용

    if not os.path.exists(args_tmp['model_path']): #model_path, sample_path 없으면 생성~
        os.makedirs(args_tmp['model_path'])
    if not os.path.exists(args_tmp['sample_path']):
        os.makedirs(args_tmp['sample_path'])

    generator = Generator(args_tmp['batchSize']) #생성모델 선언 - batch_size 정의
    discriminator = Discriminator(args_tmp['batchSize']) #분별모델 선언 - batch_size 정의

    criterionGAN = nn.BCELoss() #GAN 목적함수 - BCE
    criterionL1 = nn.L1Loss() #L1 목적함수 - L1항

    g_optimizer = optim.Adam(generator.parameters(), args_tmp['lr'], [args_tmp['beta1'], args_tmp['beta2']]) #생성모델 최적화 함수, 전역 변수 입력하여 선언
    d_optimizer = optim.Adam(discriminator.parameters(), args_tmp['lr'], [args_tmp['beta1'], args_tmp['beta2']]) #분별모델 최적화 함수, 전역 변수 입력하여 선언

    if torch.cuda.is_available(): #GPU사용 가능하면 
        print("Using GPU")
        generator = generator.cuda() #생성모델 GPU사용 변환
        discriminator = discriminator.cuda() #분별모델 gPU사용 변환

    total_step = len(data_loader) #전체 step수 저장 
    for epoch in range(args_tmp['num_epochs']): #epoch만큼 진행
        for i, (input_gray, input_ab, target) in enumerate(data_loader): #batch_size만큼 받아오면서 진행
            input_gray = input_gray.cuda()
            input_ab = input_ab.cuda()
            target = target.cuda()
            
            discriminator.zero_grad() #분별모델 gradiant 초기화

            input_gray = to_variable(input_gray) #variable로 변환
            input_ab = to_variable(input_ab) #variable로 변환
            output_ab = generator(input_gray) #가짜 이미지 생성
            target = to_variable(target) #variable로 변환

            # d_optimizer.zero_grad()

            pred_fake = discriminator(input_gray, output_ab) #가짜 이미지 분별
            #print(pred_fake)
            loss_D_fake = GAN_Loss(pred_fake, False, criterionGAN) #Loss함수 연산 - 가짜이미지 label 

            pred_real = discriminator(input_gray, input_ab) #진짜 이미지 분별
            loss_D_real = GAN_Loss(pred_real, True, criterionGAN) #Loss함수 연산 - 진짜이미지 label

            # Combined loss
            loss_D = (loss_D_fake + loss_D_real) * 0.5 #진짜이미지, 가짜이미지 Loss 합
            loss_D.backward(retain_graph=True) #오류역전파
            d_optimizer.step() #w 갱신

            generator.zero_grad() #생성모델 gradient 초기화

            pred_fake = discriminator(input_gray, output_ab) #가짜 이미지 분별
            loss_G_GAN = GAN_Loss(pred_fake, True, criterionGAN) #생성한 가짜이미지를 진짜 이미지와 GAN Loss함수 연산

            loss_G_L1 = criterionL1(output_ab, input_ab) # 생성한 가짜 ab채널과 진짜 ab채널의 거리 연산

            loss_G = loss_G_GAN + loss_G_L1 * args_tmp['lambda_A'] #Loss함수 합쳐서 연산 lambda_a만큼 L1항 영향.
            loss_G.backward() #오류역전파
            g_optimizer.step() #w갱신

            if (i + 1) % args_tmp['log_step'] == 0: #log step 되면 
                print('Epoch [%d/%d], BatchStep[%d/%d], D_Real_loss: %.4f, D_Fake_loss: %.4f, G_loss: %.4f, G_L1_loss: %.4f' #Log 출력
                      % (epoch + 1, args_tmp['num_epochs'], i + 1, total_step, loss_D_real.data, loss_D_fake.data, loss_G_GAN.data, loss_G_L1.data))

            if (i + 1) % args_tmp['sample_step'] == 0: #sample step 되면
            #if True:
                save_path = {'grayscale': 'pix2pix_results/pix2pix_gray/', 'colorized': 'pix2pix_results/pix2pix_color/', 'real': 'pix2pix_results/pix2pix_real/'}
                save_name = 'img-{}-epoch-{}.jpg'.format(i, epoch) #sample 생성
                if i+1 % 500 == 0:
                    to_rgb(input_gray.cpu(), ab_input=output_ab.detach().cpu(), real_ab=input_ab.cpu(),  save_path=save_path, save_name=save_name)
                else:
                    to_rgb(input_gray.cpu(), ab_input=output_ab.detach().cpu(), real_ab=input_ab.cpu(),  save_path=None, save_name=None)

        if epoch % args_tmp['save_freq'] == 0 or epoch == args_tmp['num_epochs']:
            g_path = f'results_{epoch + 1}' #epoch마다 
            torch.save(generator.state_dict(), g_path) #model 저장

    # img = default_loader("data/c1b.png")
    # img = np.asarray(transform_t(img).numpy())
    # lab = torch.from_numpy(img.transpose((1, 2, 0))).float()
    # img_original = rgb2gray(lab) #흑백이미지 생성
    # img_original = torch.from_numpy(img_original).unsqueeze(0).float()
    # img_use = to_variable(img_original.cuda())
    # img_use = img_use.unsqueeze(0)
    img_use, _, _ = next(iter(data_loader))

    output = generator(img_use.cuda())
    output = to_variable(output)

    save_path = {'grayscale': 'results/gray/', 'colorized': 'results/color/', 'real': 'results/real/'}
    # save_name = 'temp.jpg'
    length = len(img_use)

    for i in range(length):
        plt.clf() # clear matplotlib
        plt.subplots(nrows=1, ncols=2) #격자 구조 생성
        color_image = torch.cat((img_use[i].cpu(), output[i].cpu()), 0).numpy() # 흑백, ab채널 합치기
        color_image = color_image.transpose((1, 2, 0))  # 채널 뒤로 빼기
        color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100 
        color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   #~정규화 풀고 RGB범위 맞춰주기
        color_image = lab2rgb(color_image.astype(np.float64)) #lab에서 rgb로
        
        plt.imsave(arr=img_use[i].squeeze().numpy(), fname='{}{}'.format(save_path['grayscale'], f"temp_{i}.jpg"), cmap='gray')
        plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], f"temp_{i}.jpg"))


if __name__ == "__main__":
    main()