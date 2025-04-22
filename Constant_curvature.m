function q = Constant_curvature(p)
    % This function calculates the length changes for a soft actuator under the assumption of constant curvature.
    % The input (x, y, z) represents the target position in 3D space.
    % The outputs (q1, q2, q3) are the length changes for the three chambers.
    % Parameters
    x = p(1);
    y = p(2);
    z = p(3);
    d = 28; % Distance from the center of the chamber to the center of the base platform mm

    aa = 43.5;   % Fixed distance from the base to the cross-section of the chamber

    % Special case: If the position is at the origin (0,0), return straight lengths
    if (x == 0) && (y == 0)
        q1 = (z - aa);
        q2 = (z - aa);
        q3 = (z - aa);
        q = [q1,q2,q3];
        return;
    end
    % Calculate the angle in the XY plane
    phi = atan2(y, x);
    % Initial guess for the solver
    initialGuess = [x; y; z; 1; 0.1];
    % Define the nonlinear equations to solve
    function f = nonlinearEquations(variables)
        x0 = variables(1);
        y0 = variables(2);
        z0 = variables(3);
        k0 = variables(4);  %k0 = 1/ro
        theta0 = variables(5);
        f(1) = 2 * sqrt(x0^2 + y0^2) / (x0^2 + y0^2 + z0^2) - k0; 
        f(2) = acos(1 - k0 * sqrt(x0^2 + y0^2)) - theta0;
        f(3) = x0 + aa * sin(theta0) * cos(phi) - x;
        f(4) = y0 + aa * sin(theta0) * sin(phi) - y;
        f(5) = z0 + aa * cos(theta0) - z;
    end
    % Set options for the fsolve function
    options = optimset('MaxIter', 1e8, 'MaxFunEvals', 1e8);
    % Solve the nonlinear equations using fsolve
    solution = fsolve(@nonlinearEquations, initialGuess, options);
    % Extract real parts of the solution
    x1 = real(solution(1));
    y1 = real(solution(2));
    z1 = real(solution(3));
    k = real(solution(4));
    theta = real(solution(5));
    % Calculate the radii of curvature for each chamber
    R1 = 1 / k - d * sin(phi);
    R2 = 1 / k + d * sin(pi / 3 + phi);
    R3 = 1 / k - d * cos(pi / 6 + phi);
    % Calculate the length changes for each chamber
    q1 = double(theta * R1);
    q2 = double(theta * R2);
    q3 = double(theta * R3);
    q = [q1, q2, q3];
end