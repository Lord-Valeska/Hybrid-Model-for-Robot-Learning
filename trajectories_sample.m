% 1) 圆台参数
n_circles = 100;
z_max     = 120;    z_min     = 100;
diam_max  = 10;    diam_min  = 1;

z_lin   = linspace(z_max, z_min, n_circles);
r_lin   = linspace(diam_min/2, diam_max/2, n_circles);
r_fun   = @(zz) interp1(z_lin, r_lin, zz, 'linear', 'extrap');

% 2) 先算并画圆台表面
n_surf_z = 100;  n_surf_t = 200;
Zs = linspace(z_max, z_min, n_surf_z);
Ts = linspace(0, 2*pi, n_surf_t);
[ZZ,TT] = meshgrid(Zs,Ts);
Xs = r_fun(ZZ).*cos(TT);
Ys = r_fun(ZZ).*sin(TT);

figure; hold on; grid on; axis equal; view(3);
surf(Xs,Ys,ZZ,'FaceAlpha',0.15,'EdgeColor','none');
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
title('Continuous frustum & random trajectories');

% 3) 轨迹采样参数
step_len = 1;    n_steps = 50;

% 4) containers
trajectories = cell(n_circles,1);          % 每元 struct: p、q、angle、dAngle
big_dAngle = zeros(n_circles * (n_steps + 1), 4);% 轨迹ID | Δθ1 Δθ2 Δθ3
row_ptr      = 1;

colors = lines(n_circles);

for idx = 1:n_circles
    % 4.1 生成随机轨迹 P (51×3)
    z0  = z_lin(idx);
    P   = zeros(n_steps+1,3);  P(1,:) = [0,0,z0];
    for k=2:n_steps+1
        while true
            dir  = randn(1,3); dir = dir/norm(dir);
            cand = P(k-1,:) + step_len*dir;
            if cand(3)<z_min || cand(3)>z_max, continue; end
            if hypot(cand(1),cand(2)) <= r_fun(cand(3)), break; end
        end
        P(k,:) = cand;
    end
    
    % 4.2 按点求 q、angle
    Q   = zeros(size(P));
    Ang = zeros(size(P));
    for k=1:size(P,1)
        q = Constant_curvature(P(k,:)');    Q(k,:) = q';
        a1 = find_angle_for_length(q(1));
        a2 = find_angle_for_length(q(2));
        a3 = find_angle_for_length(q(3));
        Ang(k,1:3) = [a1(1), a2(1), a3(1)]; % 取第一个根
    end
    
    % 4.3 角度增量
    dAng = [ Ang(1, :); diff(Ang, 1, 1) ];  % 51×3
    
    % 4.4 累加进“大矩阵”以便一次写 CSV
    rows = row_ptr : row_ptr + n_steps;      % n_steps 行
    big_dAngle(rows, 1) = idx;             % 第一列存轨迹编号
    big_dAngle(rows, 2:4) = dAng;
    row_ptr = row_ptr + (n_steps+1);
    
    % 4.5 打包 & 画轨迹
    trajectories{idx} = struct('p',P,'q',Q,'angle',Ang,'dAngle',dAng);
    plot3(P(:,1),P(:,2),P(:,3),'-','Color',colors(idx,:),'LineWidth',1);
end
hold off;

% 5) 写 CSV —— 所有增量堆叠在一起
% 列: trajectory_id,  Δθ1, Δθ2, Δθ3
big_dAngle(:,2:4) = round(big_dAngle(:,2:4), 4);

% 打开文件
fid = fopen('command_all2.csv','w');

% 写入，第一列整数，其余三列保留 4 位小数
fmt = '%d,%.4f,%.4f,%.4f\n';
fprintf(fid, fmt, big_dAngle.');   % 注意转置，让 fprintf 按列读入

fclose(fid);
disp('Saved command_all.csv with 4‑decimal precision (using fprintf)');

% 6) 使用示例：读取第 10 条轨迹的增量
sel = 10;
dA  = trajectories{sel}.dAngle;        % 50×3
disp(['First Δangle of traj ',num2str(sel),': ', mat2str(dA(1,:))]);

