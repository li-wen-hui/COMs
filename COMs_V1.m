clc;clear;
image_path = '\LenaRGB.tiff';
img = imread(image_path);
img = imresize(img, [256, 256]);
T = 40; % max_order
MMT = PHFM_func_FFT(img, T, 256, 512);
I_rec = PHFM_reconstruct_func(size(img,1), MMT);
figure; subplot(1,2,1); imshow(img); subplot(1,2,2); imshow(uint8(abs(I_rec))); 

function MMT = PHFM_func_FFT(img, T, Mr, Mtheta)

    img = double(img);
    [H, W, ~] = size(img);

    t = (0:Mr-1)' / Mr;
    rho = sqrt(t);

    theta = 2*pi*(0:Mtheta-1) / Mtheta;

    [Theta, Rho] = meshgrid(theta, rho);

    Xq = Rho .* cos(Theta);
    Yq = Rho .* sin(Theta);

    x = linspace(-1, 1, W);
    y = linspace(1, -1, H);

    FR = interp2(x, y, img(:,:,1), Xq, Yq, 'linear', 0);
    FG = interp2(x, y, img(:,:,2), Xq, Yq, 'linear', 0);
    FB = interp2(x, y, img(:,:,3), Xq, Yq, 'linear', 0);

    FFT_R_theta = fft(FR, [], 2);
    FFT_G_theta = fft(FG, [], 2);
    FFT_B_theta = fft(FB, [], 2);

    m_list = -T:T;
    m_index = mod(m_list, Mtheta) + 1;

    C_R = real(FFT_R_theta(:, m_index));
    C_G = real(FFT_G_theta(:, m_index));
    C_B = real(FFT_B_theta(:, m_index));

    S_R = -imag(FFT_R_theta(:, m_index));
    S_G = -imag(FFT_G_theta(:, m_index));
    S_B = -imag(FFT_B_theta(:, m_index));

    c_mu = 1 / sqrt(2);

    A_R = C_R + c_mu * (S_B - S_G);
    A_I = C_G - c_mu * (S_R + S_B);
    A_J = C_B + c_mu * (S_R - S_G);

    FFT_R_radial = fft(A_R, [], 1);
    FFT_I_radial = fft(A_I, [], 1);
    FFT_J_radial = fft(A_J, [], 1);

    scale = 2 / (Mr * Mtheta);

    num_m = numel(m_list);
    num_moments = (T+1) * num_m;

    MMT = zeros(5, num_moments);
    cnt = 1;

    for n = 0:T

        if n == 0
            Mr_values = scale * sum(A_R, 1) / sqrt(2);
            Mi_values = scale * sum(A_I, 1) / sqrt(2);
            Mj_values = scale * sum(A_J, 1) / sqrt(2);

        elseif mod(n, 2) == 0
            q = n / 2;

            Mr_values = scale * real(FFT_R_radial(q+1, :));
            Mi_values = scale * real(FFT_I_radial(q+1, :));
            Mj_values = scale * real(FFT_J_radial(q+1, :));

        else
            q = (n + 1) / 2;

            Mr_values = scale * (-imag(FFT_R_radial(q+1, :)));
            Mi_values = scale * (-imag(FFT_I_radial(q+1, :)));
            Mj_values = scale * (-imag(FFT_J_radial(q+1, :)));
        end

        range = cnt:(cnt + num_m - 1);

        MMT(1, range) = n;
        MMT(2, range) = m_list;
        MMT(3, range) = Mr_values;
        MMT(4, range) = Mi_values;
        MMT(5, range) = Mj_values;

        cnt = cnt + num_m;
    end
end

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
