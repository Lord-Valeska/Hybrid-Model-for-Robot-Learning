% 1) 圆台参数 （用来画背景，不影响轨迹数量）
z_max    = 120;   
z_min    = 110;
diam_max = 5;    
diam_min = 5;
n_surf_z = 100;  
n_surf_t = 200;

z_lin = linspace(z_max, z_min, 2);
r_lin = linspace(diam_min/2, diam_max/2, 2);
r_fun = @(zz) interp1(z_lin, r_lin, zz, 'linear', 'extrap');

% 2) 画出 frustum 表面
Zs = linspace(z_max, z_min, n_surf_z);
Ts = linspace(0, 2*pi, n_surf_t);
[ZZ,TT] = meshgrid(Zs,Ts);
Xs = r_fun(ZZ).*cos(TT);
Ys = r_fun(ZZ).*sin(TT);

figure; hold on; grid on; axis equal; view(3);
surf(Xs,Ys,ZZ,'FaceAlpha',0.15,'EdgeColor','none');
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
title('Continuous frustum & single circular trajectory');

% 3) 圆形轨迹参数
n_steps = 20;                      
z0      = (z_max + z_min)/2;       % 轨迹所在高度，可自由选择 [z_min,z_max]
radius  = r_fun(z0) * 0.9;         % 取该高度处 frustum 半径的 90%，确保在内部

theta = linspace(0, 2*pi, n_steps+1)';  
P     = [ radius*cos(theta), ...   % X
          radius*sin(theta), ...   % Y
          z0*ones(n_steps+1,1) ];  % Z

% 4) 按点求 q、angle
Q   = zeros(size(P));
Ang = zeros(size(P));
for k = 1:size(P,1)
    q = Constant_curvature(P(k,:)');    Q(k,:)   = q';
    a1 = find_angle_for_length(q(1));
    a2 = find_angle_for_length(q(2));
    a3 = find_angle_for_length(q(3));
    Ang(k,1:3) = [a1(1), a2(1), a3(1)];
end

% 5) 角度增量
dAng = [ Ang(1, :); diff(Ang, 1, 1) ];

% 6) 写 CSV
big_dAngle = [ ones(n_steps+1,1), dAng ];
big_dAngle(:,2:4) = round(big_dAngle(:,2:4), 4);
fid = fopen('command_test.csv','w');
fprintf(fid, '%d,%.4f,%.4f,%.4f\n', big_dAngle.');
fclose(fid);
disp('Saved command_test.csv');

% 7) 画轨迹
plot3(P(:,1), P(:,2), P(:,3), 'r-', 'LineWidth', 2);