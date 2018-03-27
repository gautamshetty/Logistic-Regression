% Implementation of logistic regression.
function [ output_args ] = logistic_regression( training_file, degree, test_file )

    file_data = load (training_file);

    [r, c] = size (file_data);

    x_input = file_data (:, 1:c-1);
    target = file_data (:, c);

    target(target ~= 1) = 0;

    [r, c] = size (x_input);

    fie = [];
    for j=1:c
        for i=1:degree
            fie = [fie x_input(:,j).^i];
        end
    end

    [r, c] = size (fie);

    fie = [ones(r, 1) fie];
    fie = fie';

    weights_old = zeros(1, c + 1);

    Ew_old = 0.0;
    sum_weights = 0.001;
    while true

        y = zeros(r, 1);
        for i=1:r
            y(i, 1) = 1 / (1 + exp(weights_old(1,:) * fie(:,i)));
        end

        Ew = 0.0;
        for i=1:length(target)
            Ew = Ew + (target(i, 1) * log(y(i, 1)) + (1 - target(i, 1)) * log(1 - y(i, 1)));
        end

        Ew = -Ew;

        if (Ew - Ew_old) < 0.001 || sum_weights < 0.001
            break;
        end

        y = y';
        rnn = zeros(1,length(y));
        for i=1:length(y)
            rnn(1,i) = y(1,i) * (1 - y(1,i));
        end

        rnn = diag(rnn);

        weights_old = weights_old';
        weights_new = weights_old - pinv(fie * rnn * fie') * fie * (y' - target);

        disp (weights_old(1, 1));
        disp (weights_old(2, 1));

        weights_diff = weights_new - weights_old;
        sum_weights = sum(abs(weights_diff));

        Ew_old = Ew;

        weights_old = weights_new';

    end

    for i=1:length(weights_old)
        fprintf('w%d=%.4f\n', i, weights_old(i));
    end

    file_data = load (test_file);

    [r, c] = size (file_data);

    x_input = file_data (:, 1:c-1);
    target = file_data (:, c);

    target(target ~= 1) = 0;

    [r, c] = size (x_input);

    fie = [];
    for j=1:c
        for i=1:degree
            fie = [fie x_input(:,j).^i];
        end
    end

    [r, c] = size (fie);

    fie = [ones(r, 1) fie];
    fie = fie';

    y = zeros(r, 1);
    cum_accuracy = 0.0;
    for i=1:r

        y(i, 1) = 1 / (1 + exp(weights_old(1,:) * fie(:,i)));

        predicted_class = 1;
        probability = y(i, 1);    
        if y(i, 1) < 0.5
            predicted_class = 0;
            probability = 1 - y(i, 1);
        end

        accuracy = 1;
        if predicted_class ~= target(i)
            accuracy = 0;
        else
            if probability == 0.5
                accuracy = 0.5;
            end
        end

        cum_accuracy = cum_accuracy + accuracy;

        fprintf('ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n', (i-1), predicted_class, probability, target(i), accuracy);

    end

    classification_accuracy = cum_accuracy / r;

    fprintf('classification accuracy=%6.4f\n', cum_accuracy / r);

end