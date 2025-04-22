function solutions = find_angle_for_length(y_target)
    % find_x_for_y finds the x values within [0, 180] for a given y_target
    % that satisfy the polynomial equation in the range [0, 180].
    %
    % Usage:
    % solutions = find_x_for_y(y_target)
    %
    % Input:
    % - y_target: The target y value for which corresponding x values are needed.
    %
    % Output:
    % - solutions: Array of x values within [0, 180] that satisfy f(x) = y_target.

    % Define the polynomial function
    f = @(x) 1.327e-10*x.^5 - 7.894e-08*x.^4 + 1.314e-05*x.^3 ...
             - 0.001259*x.^2 - 0.1502*x + 80.92;

    % Define the function to find roots for f(x) = y_target
    root_function = @(x) f(x) - y_target;

    % Set the range of initial guesses
    x_range = linspace(0, 180, 100); % Initial guesses across the range [0, 180]
    solutions = []; % Initialize empty array for solutions

    % Loop over each initial guess in x_range
    for x_guess = x_range
        try
            % Use fzero to find the root near x_guess
            x_solution = fzero(root_function, x_guess);

            % Check if the solution is within the specified range and not duplicated
            if x_solution >= 0 && x_solution <= 180 && ~ismembertol(x_solution, solutions, 1e-4)
                solutions(end+1) = x_solution; % Append unique solution
            end
        catch
            % Ignore cases where fzero fails
        end
    end
end
