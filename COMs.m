clc;clear;
image_path = '\LenaRGB.tiff'; 
img = imread(image_path);
img = imresize(img, [256, 256]);
T =40; % max_order
MMT = PHFM_func(img, T);
I_rec = PHFM_reconstruct_func(size(img,1), MMT);
figure; subplot(1,2,1); imshow(img); subplot(1,2,2); imshow(uint8(abs(I_rec))); 
function I_rec = PHFM_reconstruct_func(N, MMT)
    [X, Y] = meshgrid(linspace(-1, 1, N), linspace(1, -1, N));
    [theta, r] = cart2pol(X, Y);
    idx = (r <= 1);
    R2 = r.^2;

    Rec_R = zeros(N, N);
    Rec_G = zeros(N, N);
    Rec_B = zeros(N, N);

    c_mu = 1/sqrt(2);
    num_moments = size(MMT, 2);

    for k = 1:num_moments
        n  = MMT(1, k);
        m  = MMT(2, k);
        Mr = MMT(3, k);
        Mi = MMT(4, k);
        Mj = MMT(5, k);

        if n == 0
            Tn = (1 / sqrt(2)) * ones(size(R2));
        elseif mod(n, 2) == 1
            Tn = sin((n + 1) * pi * R2);
        else
            Tn = cos(n * pi * R2);
        end
        Tn(~idx) = 0;

        cos_mt = cos(m * theta);
        sin_mt = sin(m * theta);

        Kir = Tn .* (cos_mt - c_mu * sin_mt);
        Kii = Tn .* (-c_mu * sin_mt);
        Kij = Tn .* ( c_mu * sin_mt);

        Rec_R = Rec_R + (Mr*Kir - Mi*Kij - Mj*Kii);
        Rec_G = Rec_G + (Mr*Kii + Mi*Kir - Mj*Kij);
        Rec_B = Rec_B + (Mr*Kij + Mi*Kii + Mj*Kir);
    end

    I_rec = cat(3, Rec_R, Rec_G, Rec_B);
    for c = 1:3
        tmp = I_rec(:,:,c);
        tmp(~idx) = 0;
        I_rec(:,:,c) = tmp;
    end
end
function MMT = PHFM_func(img, T)
[N, ~, ~] = size(img);
img = double(img);
[X, Y] = meshgrid(linspace(-1, 1, N), linspace(1, -1, N));
[theta, r] = cart2pol(X, Y);
idx = (r <= 1);
R2 = r.^2;

f_R = img(:,:,1); f_G = img(:,:,2); f_B = img(:,:,3);
f_R(~idx) = 0; f_G(~idx) = 0; f_B(~idx) = 0;

c_mu = 1/sqrt(2);
factor = 8 / (pi * N^2);
max_moments = (T+1)*(2*T+1);
MMT = zeros(5, max_moments);
cnt = 1;

for n = 0:T
    for m = -T:T
        if n == 0
            Tn = (1 / sqrt(2)) * ones(size(R2));
        elseif mod(n, 2) == 1
            Tn = sin((n + 1) * pi * R2);
        else
            Tn = cos(n * pi * R2);
        end
        Tn(~idx) = 0;

        cos_mt = cos(m * theta);
        sin_mt = sin(m * theta);

        Kr = Tn .* cos_mt;
        Ki = Tn .* (-c_mu * sin_mt);
        Kj = Tn .* ( c_mu * sin_mt);

        term_real = f_R.*Kr - f_G.*Kj - f_B.*Ki;
        term_i    = f_R.*Ki + f_G.*Kr - f_B.*Kj;
        term_j    = f_R.*Kj + f_G.*Ki + f_B.*Kr;

        val_real = sum(term_real(idx)) * factor;
        val_i    = sum(term_i(idx))    * factor;
        val_j    = sum(term_j(idx))    * factor;

        MMT(1, cnt) = n;
        MMT(2, cnt) = m;
        MMT(3, cnt) = val_real;
        MMT(4, cnt) = val_i;
        MMT(5, cnt) = val_j;

        cnt = cnt + 1;
    end
end

MMT = MMT(:, 1:cnt-1);
end