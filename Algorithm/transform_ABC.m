function [matZ, dbLamda, stats] = transform_ABC(matX, cvGroup,dbLamdaMin, dbLamdaMax, dbLamdaStep)

% ABC 对代谢组学数据进行自适应 Box-Cox 变换。
%
%   [matZ, LAMBDA, STATS] = ABC(matX, GROUP, LAMBDAMIN, LAMBDAMAX, LAMBDASTEPS)
%   对 matX 中的每个代谢物分别搜索最优 Box-Cox 参数，使各实验组内数据的
%   Shapiro-Wilk 正态性检验联合得分尽可能高。
%
%   输入参数
%   --------
%   matX
%       n×p 数据矩阵，其中每一行代表一个样本，每一列代表一个代谢物。
%
%   cvGroup
%       n×1 样本分组向量。ABC 在每个组内分别计算正态性检验结果，
%       再将所有组的结果合并，用于选择每个代谢物的最优 lambda。
%
%   dbLamdaMin
%       dbLamda 搜索范围的下限，缺省值为 -3。
%
%   dbLamdaMax
%       dbLamda 搜索范围的上限，缺省值为 3。
%
%   dbLamdaStep
%       dbLamda 网格搜索步长，缺省值为 0.01。
%
%   输出参数
%   --------
%   matZ
%       n×p 的 ABC 变换后数据矩阵，方向与输入矩阵 matX 相同。
%
%   dbLamda
%       1×p 向量，保存每个代谢物最终选择的 Box-Cox 参数。
%       如果某个代谢物没有进行 ABC 变换，则对应位置为 NaN。
%
%   stats
%       保存各代谢物处理结果的结构体：
%         transformed          是否执行了 ABC 变换。
%         note                 变换成功或未变换的原因。
%         originalScore        原始数据的联合正态性得分。
%         selectedScore        最终选中结果的联合正态性得分。
%
%   算法原理
%   --------
%   对第 j 个代谢物，首先将数据按最小值缩放：
%
%       x_scaled = 2*x/min(x)。
%
%   然后对 dbLamdaMin:dbLamdaStep:dbLamdaMax 中的每个 dbLamda 计算：
%
%       dbLamda = 0： z = log(x_scaled)
%       dbLamda ≠ 0： z = (x_scaled^dbLamda-1)/lambda。
%
%   程序在每个实验组内对 z 进行 Shapiro-Wilk 检验，并用下式合并各组：
%
%       score(dbLamda) = sum(log10(p_g(dbLamda)))。
%
%   如果某个 dbLamda 的得分高于原始数据，则选择得分最高的 lambda；
%   否则保留原始数据。每个代谢物均被独立处理。
%
%   本文件包含 ABC 所需的全部自定义子函数，不调用 transform_ABC.m
%   或其他项目函数。代码假定输入已经具有正确的类型、维度和参数取值，
%   因此不检查输入参数的数值类型、矩阵维度或合法范围。
%
%   完整参数调用示例
%   ----------------
%   [matZ, dbLamda, stats] = ABC(matX, cvGroup, "default", -3, 3, 0.01, 0, true);
%
%   使用全部缺省参数
%   ----------------
%   [matZ, dbLamda, stats] = ABC(matX, cvGroup);

    if ~exist('dbLamdaStep','var'),dbLamdaStep = 0.01;end
    if ~exist('dbLamdaMax','var'), dbLamdaMax  = 3;   end
    if ~exist('dbLamdaMin','var'), dbLamdaMin  = -3;  end

    [nSample, nMetabolite] = size(matX);
    if ~exist("cvGroup","var"),cvGroup = ones(nSample,1);end

    % 将分组统一为列向量。string 可以同时容纳数值、字符和分类分组标签。
    cvGroup = string(cvGroup(:));
    rvLamda = dbLamdaMin:dbLamdaStep:dbLamdaMax;

    matZ = matX;
    dbLamda = nan(1, nMetabolite);
    stats.transformed = false(1, nMetabolite);
    stats.note = strings(1, nMetabolite);
    stats.originalScore = nan(1, nMetabolite);
    stats.selectedScore = nan(1, nMetabolite);

    for j = 1:nMetabolite
        result = localOptimizeLambda(matX(:, j), cvGroup, rvLamda);

        matZ(:, j) = result.transformedData;
        dbLamda(j) = result.dbLamda;
        stats.transformed(j) = result.transformed;
        stats.note(j) = result.note;
        stats.originalScore(j) = result.originalScore;
        stats.selectedScore(j) = result.selectedScore;
    end
end


function result = localOptimizeLambda(x, cvGroup, rvLamda)
%LOCALOPTIMIZELAMBDA 搜索单个代谢物的最优 Box-Cox 参数。

    x = x(:);
    groupLevels = unique(cvGroup, "stable");
    nGroups = numel(groupLevels);

    result.transformedData = x;
    result.dbLamda = NaN;
    result.transformed = false;
    result.note = "未执行变换";
    result.originalScore = NaN;
    result.selectedScore = NaN;

    % Box-Cox 变换要求所有数值均为正数。
    minimumValue = min(x);
    if ~isfinite(minimumValue) || minimumValue <= 0
        result.note = "存在非正值，未执行 ABC 变换";
        return;
    end

    % 与原始 ABC 实现一致，将最小值缩放为 2。
    xScaled = x / minimumValue * 2;
    groupData = cell(nGroups, 1);

    % Shapiro-Wilk 检验要求每组至少 3 个样本、不超过 5000 个样本，
    % 并且组内数值不能完全相同。这些判断属于算法适用条件，而不是
    % 对函数输入参数类型或参数范围的检查。
    for g = 1:nGroups
        currentGroup = xScaled(cvGroup == groupLevels(g));
        groupData{g} = currentGroup;

        if numel(currentGroup) < 3
            result.note = "至少一个组少于 3 个样本，未执行 ABC 变换";
            return;
        end
        if numel(currentGroup) > 5000
            result.note = "至少一个组超过 5000 个样本，未执行 ABC 变换";
            return;
        end
        if all(currentGroup == currentGroup(1))
            result.note = "至少一个组的数值完全相同，未执行 ABC 变换";
            return;
        end
    end

    % 先计算未经 Box-Cox 变换时，各组 Shapiro-Wilk p 值的联合得分。
    originalPValues = zeros(nGroups, 1);
    for g = 1:nGroups
        originalPValues(g) = localShapiroWilkPValue(groupData{g});
    end
    originalScore = sum(log10(originalPValues));

    % 对给定范围内的所有 dbLamda 逐一进行网格搜索。
    lambdaScores = zeros(numel(rvLamda), 1);
    for k = 1:numel(rvLamda)
        currentLambda = rvLamda(k);
        pValues = zeros(nGroups, 1);

        for g = 1:nGroups
            transformedGroup = localBoxCox(groupData{g}, currentLambda);
            pValues(g) = localShapiroWilkPValue(transformedGroup);
        end

        lambdaScores(k) = sum(log10(pValues));
    end

    % 将原始数据也作为一个候选项。只有某个 Box-Cox 结果优于原始数据时，
    % 才真正执行变换，避免对本身已经具有最佳正态性的数据做无益处理。
    [bestScore, bestIndex] = max([lambdaScores; originalScore]);

    result.originalScore = originalScore;
    result.selectedScore = bestScore;

    if bestIndex > numel(rvLamda)
        result.note = "原始数据的正态性得分最高，保留原始数据";
        return;
    end

    selectedLambda = rvLamda(bestIndex);
    result.transformedData = localBoxCox(xScaled, selectedLambda);
    result.dbLamda = selectedLambda;
    result.transformed = true;
    result.note = "已执行 ABC 变换";
end

function z = localBoxCox(x, dbLamda)
%LOCALBOXCOX 根据 dbLamda 计算 Box-Cox 变换。

    if abs(dbLamda) < 1e-12
        z = log(x);
    else
        z = (x .^ dbLamda - 1) ./ dbLamda;
    end
end

function pValue = localShapiroWilkPValue(x)
%LOCALSHAPIROWILKPVALUE 计算 Shapiro-Wilk 正态性检验的 p 值。
%
%   该实现复现 R 的 shapiro.test 所采用的核心算法。输入样本量由上层
%   ABC 逻辑保证在 3 到 5000 之间，并且数据不为常数。

    x = sort(x(:));
    dataRange = x(end) - x(1);

    % 当数据范围非常小时进行缩放，以降低浮点计算误差。
    if dataRange < 1e-10
        x = x / dataRange;
    end

    [~, pValue] = localShapiroWilkCore(x);

    % 限制 p 值的数值范围，避免后续 log10 产生负无穷。
    pValue = min(max(pValue, realmin("double")), 1);
end

function [wStatistic, pValue] = localShapiroWilkCore(x)
%LOCALSHAPIROWILKCORE Shapiro-Wilk 检验的数值计算核心。
%
%   下面的常数和多项式系数来自 Shapiro-Wilk 检验的经典数值近似。

    n = numel(x);
    halfLength = floor(n / 2);
    weights = zeros(halfLength, 1);

    coefficientG = [-2.273, 0.459];
    coefficient1 = [0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056];
    coefficient2 = [0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633];
    coefficient3 = [0.544, -0.39978, 0.025054, -6.714e-4];
    coefficient4 = [1.3822, -0.77857, 0.062767, -0.0020322];
    coefficient5 = [-1.5861, -0.31082, -0.083751, 0.0038915];
    coefficient6 = [-0.4803, -0.082676, 0.0030302];

    nDouble = double(n);

    if n == 3
        weights(1) = 0.70710678;
    else
        adjustedN = nDouble + 0.25;
        squaredSum = 0;

        for i = 1:halfLength
            weights(i) = localNormalInverse((i - 0.375) / adjustedN);
            squaredSum = squaredSum + weights(i)^2;
        end

        squaredSum = 2 * squaredSum;
        squareRootSum = sqrt(squaredSum);
        reciprocalRootN = 1 / sqrt(nDouble);
        firstWeight = localPolynomial(coefficient1, reciprocalRootN) ...
            - weights(1) / squareRootSum;

        if n > 5
            startIndex = 3;
            secondWeight = -weights(2) / squareRootSum ...
                + localPolynomial(coefficient2, reciprocalRootN);
            scaleFactor = sqrt( ...
                (squaredSum - 2*weights(1)^2 - 2*weights(2)^2) / ...
                (1 - 2*firstWeight^2 - 2*secondWeight^2));
            weights(2) = secondWeight;
        else
            startIndex = 2;
            scaleFactor = sqrt( ...
                (squaredSum - 2*weights(1)^2) / (1 - 2*firstWeight^2));
        end

        weights(1) = firstWeight;
        for i = startIndex:halfLength
            weights(i) = weights(i) / -scaleFactor;
        end
    end

    dataRange = x(end) - x(1);

    % 计算样本均值和权重均值。
    sampleMeanScaled = sum(x / dataRange) / n;
    weightSum = 0;
    for i = 1:n
        mirrorIndex = n - i + 1;
        if i ~= mirrorIndex
            weightSum = weightSum ...
                + sign(i - mirrorIndex) * weights(min(i, mirrorIndex));
        end
    end
    weightMean = weightSum / n;

    % 计算权重方差、样本方差及二者协方差。
    weightVariance = 0;
    sampleVariance = 0;
    covariance = 0;

    for i = 1:n
        mirrorIndex = n - i + 1;
        if i ~= mirrorIndex
            centeredWeight = sign(i - mirrorIndex) ...
                * weights(min(i, mirrorIndex)) - weightMean;
        else
            centeredWeight = -weightMean;
        end

        centeredSample = x(i) / dataRange - sampleMeanScaled;
        weightVariance = weightVariance + centeredWeight^2;
        sampleVariance = sampleVariance + centeredSample^2;
        covariance = covariance + centeredWeight * centeredSample;
    end

    varianceProductRoot = sqrt(weightVariance * sampleVariance);
    oneMinusW = (varianceProductRoot - covariance) ...
        * (varianceProductRoot + covariance) ...
        / (weightVariance * sampleVariance);
    oneMinusW = min(max(oneMinusW, realmin("double")), 1);
    wStatistic = 1 - oneMinusW;

    % 样本量为 3 时使用精确的小样本近似。
    if n == 3
        pValue = 1.90985931710274 ...
            * (asin(sqrt(wStatistic)) - 1.04719755119660);
        pValue = max(pValue, 0);
        return;
    end

    transformedW = log(oneMinusW);
    logN = log(nDouble);

    % 不同样本量使用不同的均值和标准差近似公式。
    if n <= 11
        gamma = localPolynomial(coefficientG, nDouble);
        if transformedW >= gamma
            pValue = 1e-99;
            return;
        end

        transformedW = -log(gamma - transformedW);
        normalMean = localPolynomial(coefficient3, nDouble);
        normalStd = exp(localPolynomial(coefficient4, nDouble));
    else
        normalMean = localPolynomial(coefficient5, logN);
        normalStd = exp(localPolynomial(coefficient6, logN));
    end

    pValue = 1 - localNormalCdf(transformedW, normalMean, normalStd);
end

function value = localPolynomial(coefficients, x)
%LOCALPOLYNOMIAL 使用嵌套乘法计算指定系数的多项式。

    value = coefficients(1);
    if numel(coefficients) > 1
        accumulatedValue = x * coefficients(end);
        for i = numel(coefficients)-1:-1:2
            accumulatedValue = (accumulatedValue + coefficients(i)) * x;
        end
        value = value + accumulatedValue;
    end
end

function x = localNormalInverse(p)
%LOCALNORMALINVERSE 通过互补误差函数计算标准正态分布分位数。

    p = min(max(p, realmin("double")), 1-eps);
    x = -sqrt(2) * erfcinv(2*p);
end

function p = localNormalCdf(x, mu, sigma)
%LOCALNORMALCDF 通过误差函数计算正态分布累积分布函数。

    standardizedValue = (x-mu) ./ (sigma*sqrt(2));
    p = 0.5 * erfc(-standardizedValue);
end
