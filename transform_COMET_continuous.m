function [matZ, stats] = transform_COMET_continuous(matX, cvY, dbLamda, dbMinGapRatio, ...
    nMaxIteration, maxFunctionEvaluations, optimalityTolerance)

% COMET_CONTINUOUS 用于连续表型的 COMET 单调变换。
%
%   [matZ, STATS] = COMET_CONTINUOUS(matX, cvY, LAMBDA, MINGAPRATIO, ...
%       MAXITERATIONS, MAXFUNCTIONEVALUATIONS, OPTIMALITYTOLERANCE)
%   对 matX 中的每一列代谢物分别进行单调变换，在保持样本丰度排序不变的
%   条件下，尽可能增强变换后代谢物与连续表型 Y 的 Pearson 相关性。
%
%   输入参数
%   --------
%   matX
%       n×p 数据矩阵，其中每一行代表一个样本，每一列代表一个代谢物。
%
%   cvY
%       n×1 连续表型向量。
%
%   dbLamda
%       变换偏移惩罚项的强度。dbLambda 越大，变换后的数值越接近原始数据；
%       dbLamda 越小，COMET 对特征-表型响应曲线的调整幅度越大。
%       缺省值 0.1
%
%   dbMinGapRatio
%       相邻变换节点之间允许的最小距离比例。该距离等于原始非重复数值
%       最小间隔乘以 dbMinGapRatio。例如，0.01 表示原始最小间隔的 1%。
%       设置正的最小距离可以保证变换函数严格单调递增。
%       缺省值 0.01
%
%   nMaxIteration
%       每次调用 fminunc 时允许的最大迭代次数。
%       缺省值 400
%
%   maxFunctionEvaluations
%       每次调用 fminunc 时允许的最大目标函数计算次数。
%       缺省值 '100*numberOfVariables'
%
%   optimalityTolerance
%       fminunc 使用的一阶最优性容差。
%       缺省值 1e-9
%
%   输出参数
%   --------
%   matZ
%       n×p 的 COMET 变换后数据矩阵。
%
%   stats
%       保存各代谢物变换结果的结构体，每个字段包含以下信息：
%         rvRawPCC        matX(:,j) 与 cvY 之间的 Pearson 相关系数。
%         rvPCC           matZ(:,j) 与 cvY 之间的 Pearson 相关系数。
%         rvSpearman      matZ(:,j) 与 matX(:,j) 之间的 Spearman 相关系数。
%         rvDeltaNorm     matZ(:,j)-matX(:,j) 的欧氏范数，即绝对变换幅度。
%         rvRelativeNorm
%                         与目标函数一致的相对扰动项：
%                         ||z-x||^2 / ||x-mean(x)||^2。
%         rvDirection      COMET 最终选择的相关方向，取值为 +1 或 -1。
%         rvLoss     最终被选中解的目标函数值。
%         arrKnotX          原始代谢物的非重复数值，即变换前的节点。
%         arrKnotZ          优化后的节点数值。
%
%   数学目标
%   --------
%   对于第 j 个代谢物，COMET 估计一个严格单调递增函数 q_j：
%
%       最大化  |corr(q_j(x_j), cvY)|
%               - dbLamda * ||q_j(x_j)-x_j||^2 / ||x_j-mean(x_j)||^2。
%
%   为了优化相关系数的绝对值，程序分别针对正相关方向和负相关方向求解
%   一次，然后保留目标函数值更小的解。
%
%   本实现假定 matX 和 cvY 已经是数值型、均为有限值、方向正确，并且样本已经
%   一一对应。代码不检查输入类型、矩阵维度、参数范围或缺失值。若某个
%   代谢物为常数，则无法估计其单调响应曲线，因此直接返回原始数值。
%
%   调用示例
%   --------
%   [matZ, stats] = COMET_continuous(matX, cvY, 0.1, 0.01, 500, 5000, 1e-9);

    cvY = cvY(:);
    [~, nMetabolite] = size(matX);

    % 所有代谢物使用同一个连续表型，因此只需对 cvY 做一次中心化。
    % Pearson 相关系数等于两个中心化向量夹角的余弦值。
    yCentered = cvY - mean(cvY);
    yNorm = norm(yCentered);

    matZ = zeros(size(matX));

    stats.rvRawPCC    = nan(1, nMetabolite);
    stats.rvPCC       = nan(1, nMetabolite);
    stats.rvSpearman  = nan(1, nMetabolite);
    stats.rvDeltaNorm = zeros(1, nMetabolite);
    stats.rvRelativeNorm = zeros(1, nMetabolite);
    stats.rvDirection = zeros(1, nMetabolite);
    stats.rvLoss      = nan(1, nMetabolite);
    stats.arrKnotX    = cell(1, nMetabolite);
    stats.arrKnotZ    = cell(1, nMetabolite);

    % 优化过程中的主要参数全部由函数输入提供。
    % 将 Display 固定为 off，避免每个代谢物优化时反复打印迭代信息。
    options = optimoptions('fminunc', ...
        'Algorithm', 'quasi-newton', ...
        'Display', 'off', ...
        'MaxIterations', nMaxIteration, ...
        'MaxFunctionEvaluations', maxFunctionEvaluations, ...
        'OptimalityTolerance', optimalityTolerance);

    for j = 1:nMetabolite
        x = matX(:, j);

        % inverseIndex 记录每个样本对应哪个非重复丰度节点。
        % 因此，原始数据中丰度相同的样本在变换后仍具有相同数值。
        [xUnique, ~, inverseIndex] = unique(x, 'sorted');

        % 常数代谢物没有可估计的响应形态，直接使用恒等变换并返回原值。
        if isscalar(xUnique)
            matZ(:, j) = x;
            stats.arrKnotX{j} = xUnique;
            stats.arrKnotZ{j} = xUnique;
            continue;
        end

        % 只在排序后的非重复 x 节点上表示变换函数 q。
        % 相邻 q 节点之间保持正增量，可保证变换严格递增并保留样本排序。
        minGap = dbMinGapRatio * min(diff(xUnique));

        % 将 q 初始化在恒等变换 q(x)=x 附近。
        % 第一个参数表示第一个节点，其余无约束参数通过 softplus 函数
        % 转换为正增量，从而避免在 fminunc 中显式设置单调约束。
        originalIncrements = diff(xUnique);
        theta0 = [xUnique(1); ...
            localSoftplusInverse(originalIncrements - minGap)];

        xCentered = x - mean(x);
        xVariation = sum(xCentered .^ 2);

        % 由于目标是 Pearson 相关系数的绝对值，因此分别拟合正、负两个方向。
        % 无论选择哪个相关方向，变换函数 q 本身始终保持单调递增。
        positiveObjective = @(theta) localObjective(theta, 1, ...
            inverseIndex, x, xVariation, yCentered, yNorm, dbLamda, minGap);
        negativeObjective = @(theta) localObjective(theta, -1, ...
            inverseIndex, x, xVariation, yCentered, yNorm, dbLamda, minGap);

        [thetaPositive, valuePositive] = fminunc( ...
            positiveObjective, theta0, options);
        [thetaNegative, valueNegative] = fminunc( ...
            negativeObjective, theta0, options);

        if valuePositive <= valueNegative
            bestTheta = thetaPositive;
            stats.rvDirection(j) = 1;
            stats.rvLoss(j) = valuePositive;
        else
            bestTheta = thetaNegative;
            stats.rvDirection(j) = -1;
            stats.rvLoss(j) = valueNegative;
        end

        arrKnotZ = localThetaToKnots(bestTheta, minGap);
        z = arrKnotZ(inverseIndex);
        matZ(:, j) = z;

        stats.rvRawPCC(j) = corr(x, cvY, 'Type', 'Pearson');
        stats.rvPCC(j) = corr(z, cvY, 'Type', 'Pearson');
        stats.rvSpearman(j) = corr(z, x, 'Type', 'Spearman');
        stats.rvDeltaNorm(j) = norm(z - x);
        stats.rvRelativeNorm(j) = sum((z - x) .^ 2) / xVariation;
        stats.arrKnotX{j} = xUnique;
        stats.arrKnotZ{j} = arrKnotZ;
    end
end

function rvLoss = localObjective(theta, rvDirection, inverseIndex, x, ...
    xVariation, yCentered, yNorm, dbLamda, minGap)

%LOCALOBJECTIVE 计算供 fminunc 最小化的负惩罚相关目标函数。

    arrKnotZ = localThetaToKnots(theta, minGap);
    z = arrKnotZ(inverseIndex);
    zCentered = z - mean(z);

    pearsonCorrelation = dot(zCentered, yCentered) / ...
        (norm(zCentered) * yNorm);

    % 使用原始特征的总变异对偏移惩罚项进行归一化，使 dbLamda 在不同丰度
    % 尺度的代谢物之间具有更好的可比性。
    displacementPenalty = sum((z - x) .^ 2) / xVariation;

    rvLoss = -(rvDirection * pearsonCorrelation ...
        - dbLamda * displacementPenalty);
end

function knots = localThetaToKnots(theta, minGap)
%LOCALTHETATOKNOTS 将无约束参数转换为严格单调递增的节点。

    firstKnot = theta(1);
    positiveIncrements = localSoftplus(theta(2:end)) + minGap;
    knots = [firstKnot; firstKnot + cumsum(positiveIncrements)];
end

function value = localSoftplus(u)
%LOCALSOFTPLUS 以数值稳定的方式计算 log(1+exp(u))。

    value = log1p(exp(-abs(u))) + max(u, 0);
end

function value = localSoftplusInverse(cvY)
%LOCALSOFTPLUSINVERSE 计算正数 cvY 对应的 softplus 反函数。

    value = cvY;
    useExactFormula = cvY <= 20;
    value(useExactFormula) = log(expm1(cvY(useExactFormula)));
end
