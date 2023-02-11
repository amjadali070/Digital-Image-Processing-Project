function varargout = DIP_Project(varargin)
% DIPASSIGN2 MATLAB code for DIPAssign2.fig
%      DIPASSIGN2, by itself, creates a new DIPASSIGN2 or raises the existing
%      singleton*.
%
%      H = DIPASSIGN2 returns the handle to a new DIPASSIGN2 or the handle to
%      the existing singleton*.
%
%      DIPASSIGN2('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALL BACK in DIPASSIGN2.M with the given input arguments.
%
%      DIPASSIGN2('Property','Value',...) creates a new DIPASSIGN2 or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before DIPAssign2_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to DIPAssign2_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help DIPAssign2

% Last Modified by GUIDE v2.5 21-May-2022 17:33:32

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @DIP_Project_OpeningFcn, ...
                   'gui_OutputFcn',  @DIP_Project_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

%====================================
%global variable for histogram
function setGlobalx(val)
global x
x = val;

function r = getGlobalx
global x
r = x;
%====================================
function setGlobalFile(val)
global p
p = val;
function z = getGlobalFile
global p
z = p;
%====================================

% --- Executes just before DIPAssign2 is made visible.
function DIP_Project_OpeningFcn(hObject, ~, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to DIPAssign2 (see VARARGIN)

% Choose default command line output for DIPAssign2
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes DIPAssign2 wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = DIP_Project_OutputFcn(~, ~, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function Upload_image_Callback(hObject, eventdata, handles)
my_image=uigetfile('*.*');
setGlobalFile(my_image);
my_image=imread(my_image);
axes(handles.axes1);
imshow(my_image);
setappdata(0,'my_image',my_image);


function Gray_Scale_Callback(hObject, eventdata, handles)
my_image=getappdata(0,'my_image');
agray=rgb2gray(my_image);
axes(handles.axes2);
imshow(agray);


% --- Executes on button press in Histrogram.
function Histrogram_Callback(hObject, eventdata, handles)
my_image=getappdata(0,'my_image');
my_image=rgb2gray(my_image);
axes(handles.axes2);
imhist(my_image);


% --- Executes on button press in reset.
function reset_Callback(hObject, eventdata, handles)
my_image=getappdata(0,'my_image');
axes(handles.axes2);
imshow(my_image);


% --- Executes on button press in Hist_Match.
function Hist_Match_Callback(hObject, eventdata, handles)
A = imread('concordaerial.png');
Ref = imread('concordorthophoto.png');
size(A);
size(Ref);
B = imhistmatch(A,Ref);
imshow(A)
title('RGB Image with Color Cast')
axes(handles.axes1);
imshowpair(A,Ref,'montage')
title('Reference Grayscale Image')
axes(handles.axes2);
imshow(B);




% --- Executes on button press in Negative.
function Negative_Callback(hObject, eventdata, handles)
my_image=getappdata(0,'my_image');
L = 2 ^ 8;                     
neg = (L - 1) - my_image;
axes(handles.axes2);
imshow(neg);


% --- Executes on button press in interpolation.
function interpolation_Callback(hObject, eventdata, handles)
  my_image=getappdata(0,'my_image');
  biLinearInter=imresize(my_image,0.3,'bilinear'); 
  
  my_image=getappdata(0,'my_image');
  nearInter=imresize(my_image,0.3,'nearest');
  
  my_image=getappdata(0,'my_image');
  bcInter=imresize(my_image,0.3,'bicubic');
  
  figure
  set(gcf,'position',get(0,'ScreenSize'));
  subplot(2,2,1),imshow(my_image,[]),title('Original Image');
  subplot(2,2,2),imshow(biLinearInter,[]),title('Bilinear');
  subplot(2,2,3),imshow(nearInter,[]),title('Nearest');
  subplot(2,2,4),imshow(bcInter,[]),title('Bicubic');
      

% --- Executes on button press in Frequency_Domain.
function Frequency_Domain_Callback(hObject, eventdata, handles)
my_image=getappdata(0,'my_image');
        input_image = rgb2gray(my_image);
        [M,N] = size(input_image);
        FT_img = fft2(double(input_image));
        D0 = 30;
        u = 0:(M-1);
        idx = find(u>M/2);
        u(idx) = u(idx)-M;
        v = 0:(N-1);
        idy = find(v>N/2);
        v(idy) = v(idy)-N;
        [V, U] = meshgrid(v, u);
        D = sqrt(U.^2+V.^2);
        H = double(D<=D0);
        G = H.*FT_img;
        output_lps = real(ifft2(double(G)));  
        
        
        [M, N] = size(input_image);
        FT_img = fft2(double(input_image));
        D0 = 10;
        u = 0:(M-1);
        idx = find(u>M/2);
        u(idx) = u(idx)-M;
        v = 0:(N-1);
        idy = find(v>N/2);
        v(idy) = v(idy)-N;
        [V, U] = meshgrid(v, u);
        D = sqrt(U.^2+V.^2);
        H = double(D > D0);
        G = H.*FT_img;
        output_hps = real(ifft2(double(G)));
        
        
        [M, N] = size(input_image);
        FT_img = fft2(double(input_image));
        n = 2; 
        D0 = 20;
        u = 0:(M-1);
        v = 0:(N-1);
        idx = find(u > M/2);
        u(idx) = u(idx) - M;
        idy = find(v > N/2);
        v(idy) = v(idy) - N;
        [V, U] = meshgrid(v, u);
        D = sqrt(U.^2 + V.^2);
        H = 1./(1 + (D./D0).^(2*n));
        G = H.*FT_img;
        output_butterlps = real(ifft2(double(G)));
        
        [M, N] = size(input_image);
        FT_img = fft2(double(input_image));
        n = 2;
        D0 = 20;
        u = 0:(M-1);
        v = 0:(N-1);
        idx = find(u > M/2);
        u(idx) = u(idx) - M;
        idy = find(v > N/2);
        v(idy) = v(idy) - N;
        [V, U] = meshgrid(v, u);
        D = sqrt(U.^2 + V.^2);
        H = 1./(1 + (D./D0).^(2*n));
        G = H.*(1-FT_img);
        output_butterhps  = real(ifft2(double(G)));
        
  figure
  set(gcf,'position',get(0,'ScreenSize'));
  subplot(2,2,1),imshow(output_lps,[]),title('Ideal Low pass');
  subplot(2,2,2),imshow(output_hps,[]),title('Ideal High pass');
  subplot(2,2,3),imshow(output_butterlps,[]),title('Butterworth Low Pass');
  subplot(2,2,4),imshow(output_butterhps,[]),title('Butterworth High Pass');

  
  
  
  


% --- Executes on button press in Filters.
function Filters_Callback(hObject, eventdata, handles)
 my_image=getappdata(0,'my_image');
        img=my_image;
        GaussianFilter = imgaussfilt(img,2);
        img1 = img-GaussianFilter; 
        img2 = img+img1;
       
        lap_img1=my_image;
        lap_img2 = fspecial('laplacian',0);
        lap_img1=im2double(lap_img1);
        lap_img3=imfilter(lap_img1,lap_img2,'replicate');
        lap_img4=lap_img1-lap_img3;
        
        my_image=rgb2gray(my_image);
        BoxFilter = conv2(single(my_image), ones(3)/9, 'same');
        
        
   
 figure
  set(gcf,'position',get(0,'ScreenSize'));
  subplot(2,2,1),imshow(img2,[]),title('Gaussian Filter');
  subplot(2,2,2),imshow(lap_img4,[]),title('Laplacian Filter');
  subplot(2,2,3),imshow(BoxFilter,[]),title('Box Filter');
  subplot(2,2,4),imshow(my_image,[]),title('Weighted Filter');
  
 
% --- Executes on button press in Blur_Noise.
function Blur_Noise_Callback(hObject, eventdata, handles)
my_image=getappdata(0,'my_image');

PSF = fspecial('motion',21,11);
Idouble = im2double(my_image);
blurred = imfilter(Idouble,PSF,'conv','circular');

wnr1 = deconvwnr(blurred,PSF);

noise_mean = 0;
noise_var = 0.0001;
blurred_noisy = imnoise(blurred,'gaussian',noise_mean,noise_var);

wnr2 = deconvwnr(blurred_noisy,PSF);

signal_var = var(Idouble(:));
NSR = noise_var / signal_var;
wnr3 = deconvwnr(blurred_noisy,PSF,NSR);

figure
  set(gcf,'position',get(0,'ScreenSize'));
  subplot(2,2,1),imshow(blurred,[]),title('Blurred Image');
  subplot(2,2,2),imshow(blurred_noisy,[]),title('Blurred and Noisy Image');
  subplot(2,2,3),imshow(wnr2,[]),title('Restoration of Blurred Noisy Image');
  subplot(2,2,4),imshow(wnr3,[]),title('Restored Blurred Image');
 
  
  

function Median_filter_Callback(hObject, eventdata, handles)
my_image=getappdata(0,'my_image');

J = imnoise(my_image,'salt & pepper',0.02);
Kaverage = filter2(fspecial('average',3),J)/255;
Kmedian = medfilt2(J);


figure
  set(gcf,'position',get(0,'ScreenSize'));
  subplot(2,2,1),imshow(my_image,[]),title('Original Image');
  subplot(2,2,2),imshow(J,[]),title('Salt & pepper Noise');
  subplot(2,2,3),imshow(Kaverage,[]),title('average');
  subplot(2,2,4),imshow(Kmedian,[]),title('Median Filter applied');
  
  
  
  


% --- Executes on button press in rotation.
function rotation_Callback(hObject, eventdata, handles)
my_image=getappdata(0,'my_image');
rotate_180 = imrotate(my_image,-180,'bilinear','crop');
rotate_260 = imrotate(my_image,-260,'bilinear','crop');
rotate_90 = imrotate(my_image,-90,'bilinear','crop');


figure
  set(gcf,'position',get(0,'ScreenSize'));
  subplot(2,2,1),imshow(my_image,[]),title('Original Image');
  subplot(2,2,2),imshow(rotate_180 ,[]),title('- 180 Rotation');
  subplot(2,2,3),imshow(rotate_260 ,[]),title('- 260 Rotation');
  subplot(2,2,4),imshow(rotate_90 ,[]),title('- 90 Rotation');
  
  

function operators_Callback(hObject, eventdata, handles)
my_image=getappdata(0,'my_image');

[~,threshold] = edge(my_image,'sobel');
fudgeFactor = 0.5;
BWs = edge(my_image,'sobel',threshold * fudgeFactor);

se90 = strel('line',3,90);
se0 = strel('line',3,0);
BWsdil = imdilate(BWs,[se90 se0]);

seD = strel('diamond',1);
BWfinal = imerode(BWsdil,seD);
BWfinal = imerode(BWfinal,seD);

figure
  set(gcf,'position',get(0,'ScreenSize'));
  subplot(2,2,1),imshow(my_image,[]),title('Original Image');
  subplot(2,2,2),imshow(BWs,[]),title('Binary Gradient Mask (Sobel)');
  subplot(2,2,3),imshow(BWsdil,[]),title('Dilated Gradient Mask (Line)');
  subplot(2,2,4),imshow(BWfinal,[]),title('Segmented Image (Diamond)');


% --- Executes on button press in restoration.
function restoration_Callback(hObject, eventdata, handles)
my_image=getappdata(0,'my_image');
mask_size = 3;
sp_cktb = my_image;
med_filter = @(mask)median(mask(:));
med_0 = sp_cktb;
med_1 = nlfilter(med_0,[mask_size,mask_size],med_filter);
med_2 = nlfilter(med_1,[mask_size,mask_size],med_filter);
med_3 = nlfilter(med_2,[mask_size,mask_size],med_filter);

figure
    set(gcf,'position',get(0,'ScreenSize'));
    subplot(2,2,1),imshow(med_0,[]),title('salt&pepper noise');
    subplot(2,2,2),imshow(med_1,[]),title('median filter for one time');
    subplot(2,2,3),imshow(med_2,[]),title('median filter for two times');
    subplot(2,2,4),imshow(med_3,[]),title('median filter for three times');
