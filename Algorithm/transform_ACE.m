function [matZ, stats] = transform_ACE(matX, cvY, dbSpan, nMaxIteration, dbTolerance)
% Held-out feature scores: apply_map(x_train,z_train,x_new).
% Uses linear interpolation inside training support and endpoint clamping outside.
% Clamping may create held-out boundary ties; fitted training nodes are unchanged.

% ACE 对每个代谢特征分别执行 bivariate Alternating Conditional Expectations。
%
%   [matZ, STATS] = transform_ACE(matX, cvY, SPAN, MAXITERATION, TOLERANCE)
%
%   对 matX 中的每一列代谢物 x_j，分别估计两个非参数变换
%
%       phi_j(x_j)  和  theta_j(Y)
%
%   通过交替条件期望迭代，使
%
%       corr(phi_j(x_j), theta_j(Y))
%
%   尽可能增大。与 COMET 不同，标准 ACE 同时允许变换代谢特征和表型，
%   且不施加单调性、秩保持或原始丰度扰动惩罚。因此，本实现适合作为
%   高自由度 optimal-transformation 参考，而不是与 COMET 完全同约束的算法。
%
%   本函数采用 feature-wise bivariate ACE：每个代谢物独立与同一个连续表型
%   cvY 进行 ACE 拟合。条件期望由一个无额外工具箱依赖的 fixed-span
%   local-linear smoother 估计；窗口按排序后的非重复 predictor 水平定义，
%   并以每个水平的样本数作为权重。相同 predictor 取值先合并，因此同值
%   样本得到相同的条件期望估计。
%
%   输入参数
%   --------
%   matX
%       n×p 数据矩阵，其中每一行代表一个样本，每一列代表一个代谢物。
%
%   cvY
%       n×1 连续表型向量。
%
%   dbSpan
%       条件期望平滑窗口占 predictor 非重复取值数的比例。
%       例如 0.20 表示每个局部窗口约覆盖 20% 的非重复 predictor 水平。
%       缺省值 0.20。
%
%   nMaxIteration
%       ACE 交替条件期望的最大迭代次数。
%       缺省值 50。
%
%   dbTolerance
%       连续两次 theta(Y) 更新的均方根差异收敛阈值。
%       缺省值 1e-6。
%
%   输出参数
%   --------
%   matZ
%       n×p 的 ACE 特征变换结果。为了使不同算法输出尺度更便于比较，
%       每个 ACE 变换后的特征最终仅通过正仿射变换匹配原始 x_j 的均值和
%       标准差；该重标度不改变相关系数或样本排序。
%
%   stats
%       保存各代谢物 ACE 结果的结构体：
%
%         rvRawPCC
%             原始 x_j 与原始 cvY 的 Pearson 相关。
%
%         rvPCC
%             ACE 变换后 matZ(:,j) 与原始 cvY 的 Pearson 相关。
%             注意：标准 ACE 的优化目标不是这一相关，而是下面的
%             rvACECorrelation。
%
%         rvACECorrelation
%             corr(phi_j(x_j), theta_j(Y))，即 bivariate ACE 的最大相关估计。
%
%         rvSpearman
%             matZ(:,j) 与原始 x_j 的 Spearman 相关。ACE 不保证该值为 1。
%
%         rvYTransformPCC / rvYTransformSpearman
%             ACE 学到的 theta_j(Y) 与原始 Y 的 Pearson / Spearman 相关，
%             用于量化 ACE 对 phenotype scale 本身的改变。
%
%         rvDeltaNorm
%             ||matZ(:,j)-matX(:,j)||_2。
%
%         rvRelativeNorm
%             ||matZ(:,j)-matX(:,j)||_2^2 /
%             ||matX(:,j)-mean(matX(:,j))||_2^2。
%
%         rvIterations
%             实际 ACE 迭代次数。
%
%         converged
%             是否在 nMaxIteration 内满足 dbTolerance。
%
%         note
%             当前特征的处理状态。
%
%         arrKnotX / arrKnotZ
%             原始 x 的排序非重复节点与这些节点对应的 ACE 特征变换值。
%
%         arrKnotY / arrKnotThetaY
%             原始 Y 的排序非重复节点与对应的 ACE theta(Y) 节点。
%
%   数学原理
%   --------
%   对单个代谢物 x，bivariate ACE 交替计算
%
%       phi(x)   <- E[theta(Y) | x]
%       theta(Y) <- E[phi(x) | Y]
%
%   并对 theta(Y) 做中心化和标准化，直至收敛。对于 p=1 的情形，
%   ACE 估计的是 X 与 Y 的 maximal correlation 变换。
%
%   重要说明
%   --------
%   1. ACE 同时学习 X 和 Y 的变换，因此其 rvACECorrelation 不能与
%      COMET 的 corr(COMET(X), raw Y) 机械等同。
%   2. ACE 不保证单调，也不保证秩保持。
%   3. 本函数不进行训练/测试拆分。若用于预测或样本外评价，应只在训练集
%      学习节点映射，再外推至验证集。
%   4. 本实现假定 matX 和 cvY 已经是数值型有限值，并且样本一一对应。
%
%   参考
%   ----
%   Breiman, L.; Friedman, J. H. Estimating Optimal Transformations for
%   Multiple Regression and Correlation. JASA 1985, 80, 580-598.
%
%   调用示例
%   --------
%   [matZ, stats] = transform_ACE(matX, cvY);
%   [matZ, stats] = transform_ACE(matX, cvY, 0.20, 50, 1e-6);

    if ~exist('dbTolerance','var')
        dbTolerance = 1e-6;
    end
    if ~exist('nMaxIteration','var')
        nMaxIteration = 50;
    end
    if ~exist('dbSpan','var')
        dbSpan = 0.20;
    end

    cvY = cvY(:);
    [nSample, nMetabolite] = size(matX);

    matZ = matX;

    stats.transformed       = false(1, nMetabolite);
    stats.rvRawPCC          = nan(1, nMetabolite);
    stats.rvPCC             = nan(1, nMetabolite);
    stats.rvACECorrelation  = nan(1, nMetabolite);
    stats.rvSpearman        = nan(1, nMetabolite);
    stats.rvYTransformPCC   = nan(1, nMetabolite);
    stats.rvYTransformSpearman = nan(1, nMetabolite);
    stats.rvDeltaNorm       = zeros(1, nMetabolite);
    stats.rvRelativeNorm    = zeros(1, nMetabolite);
    stats.rvIterations      = zeros(1, nMetabolite);
    stats.converged         = false(1, nMetabolite);
    stats.note              = strings(1, nMetabolite);
    stats.arrKnotX          = cell(1, nMetabolite);
    stats.arrKnotZ          = cell(1, nMetabolite);
    stats.arrKnotY          = cell(1, nMetabolite);
    stats.arrKnotThetaY     = cell(1, nMetabolite);

    % 所有特征共享同一个表型。若表型为常数，则 ACE 无法定义。
    yStd = std(cvY, 0);
    if ~isfinite(yStd) || yStd <= eps(max(1, abs(mean(cvY))))
        for j = 1:nMetabolite
            x = matX(:,j);
            stats.rvRawPCC(j) = localSafePearson(x, cvY);
            stats.rvPCC(j) = stats.rvRawPCC(j);
            stats.rvSpearman(j) = localSafeSpearman(x, x);
            stats.note(j) = "表型为常数，未执行 ACE 变换";
            [xUnique,~,~] = unique(x,'sorted');
            stats.arrKnotX{j} = xUnique;
            stats.arrKnotZ{j} = xUnique;
        end
        return;
    end

    for j = 1:nMetabolite
        x = matX(:, j);
        [xUnique, ~, inverseX] = unique(x, 'sorted');

        stats.rvRawPCC(j) = localSafePearson(x, cvY);

        % 常数代谢物无法学习特征变换，直接保留原值。
        if isscalar(xUnique) || std(x,0) <= eps(max(1,abs(mean(x))))
            matZ(:,j) = x;
            stats.rvPCC(j) = stats.rvRawPCC(j);
            stats.rvSpearman(j) = localSafeSpearman(x, x);
            stats.note(j) = "代谢物为常数，未执行 ACE 变换";
            stats.arrKnotX{j} = xUnique;
            stats.arrKnotZ{j} = xUnique;
            continue;
        end

        % ACE 以标准化后的原始 Y 作为 theta(Y) 初始化。
        theta = localStandardize(cvY);
        converged = false;
        nIterationUsed = 0;

        for iter = 1:nMaxIteration
            nIterationUsed = iter;

            % phi(x) <- E[theta(Y)|x]
            phi = localConditionalExpectation(x, theta, dbSpan);
            phi = phi - mean(phi);

            if std(phi,0) <= 1e-12
                break;
            end

            % theta(Y) <- E[phi(x)|Y]，再按 ACE 约束进行标准化。
            thetaNew = localConditionalExpectation(cvY, phi, dbSpan);
            thetaNew = localStandardize(thetaNew);

            if std(thetaNew,0) <= 1e-12
                break;
            end

            % ACE 解的整体符号并不唯一。固定相邻迭代的符号，避免 ± 解翻转。
            if dot(thetaNew, theta) < 0
                thetaNew = -thetaNew;
            end

            change = norm(thetaNew - theta) / sqrt(nSample);
            theta = thetaNew;

            if change <= dbTolerance
                converged = true;
                break;
            end
        end

        % 用最终 theta 重新计算一次 phi。
        phi = localConditionalExpectation(x, theta, dbSpan);
        phi = phi - mean(phi);

        if std(phi,0) <= 1e-12 || any(~isfinite(phi))
            matZ(:,j) = x;
            stats.rvPCC(j) = stats.rvRawPCC(j);
            stats.rvSpearman(j) = localSafeSpearman(x, x);
            stats.rvIterations(j) = nIterationUsed;
            stats.converged(j) = false;
            stats.note(j) = "ACE 条件期望退化为常数，保留原始数据";
            stats.arrKnotX{j} = xUnique;
            stats.arrKnotZ{j} = xUnique;
            continue;
        end

        % 固定 ACE 解的整体符号，使 theta(Y) 与原始 Y 尽量保持正方向。
        % 该操作同时翻转 phi 与 theta，不改变 ACE maximal correlation。
        thetaRawCorrelation = localSafePearson(theta, cvY);
        if isfinite(thetaRawCorrelation) && thetaRawCorrelation < 0
            theta = -theta;
            phi = -phi;
        end

        % 输出特征仅做正仿射重标度，使均值和标准差与原始 x 相同。
        phiStandard = localStandardize(phi);
        z = mean(x) + std(x,0) * phiStandard;

        matZ(:,j) = z;

        stats.transformed(j) = true;
        stats.rvPCC(j) = localSafePearson(z, cvY);
        stats.rvACECorrelation(j) = abs(localSafePearson(phi, theta));
        stats.rvSpearman(j) = localSafeSpearman(z, x);
        stats.rvYTransformPCC(j) = localSafePearson(theta, cvY);
        stats.rvYTransformSpearman(j) = localSafeSpearman(theta, cvY);
        stats.rvDeltaNorm(j) = norm(z - x);

        xVariation = sum((x - mean(x)).^2);
        if xVariation > 0
            stats.rvRelativeNorm(j) = sum((z - x).^2) / xVariation;
        else
            stats.rvRelativeNorm(j) = 0;
        end

        stats.rvIterations(j) = nIterationUsed;
        stats.converged(j) = converged;

        if converged
            stats.note(j) = "ACE 已收敛";
        else
            stats.note(j) = "ACE 达到最大迭代或局部平滑退化";
        end

        % 保存训练样本上的节点映射，便于后续诊断或外推。
        zUnique = accumarray(inverseX, z, [], @mean);
        stats.arrKnotX{j} = xUnique;
        stats.arrKnotZ{j} = zUnique;

        [yUnique, ~, inverseY] = unique(cvY, 'sorted');
        thetaUnique = accumarray(inverseY, theta, [], @mean);
        stats.arrKnotY{j} = yUnique;
        stats.arrKnotThetaY{j} = thetaUnique;
    end
end


function fitted = localConditionalExpectation(predictor, response, dbSpan)
%LOCALCONDITIONALEXPECTATION
%   用 fixed-span local-linear smoother 估计 E[response | predictor]。
%
%   1. 相同 predictor 水平先合并；
%   2. 在排序后的非重复 predictor 水平上取局部对称窗口；
%   3. 以每个水平的样本数为权重，在窗口内拟合
%
%          response = intercept + slope * predictor；
%
%   4. 在当前 predictor 节点评价局部线性拟合。
%
%   所需加权充分统计量通过累积和一次性计算，因此单次平滑近似 O(n)。

    predictor = predictor(:);
    response = response(:);

    [predictorUnique, ~, inverseIndex] = unique(predictor, 'sorted');
    nUnique = numel(predictorUnique);

    groupCount = accumarray(inverseIndex, 1, [nUnique,1], @sum);
    groupSumY = accumarray(inverseIndex, response, [nUnique,1], @sum);

    % 至少覆盖 3 个非重复水平；窗口大小尽量保持奇数以便左右对称。
    nWindow = max(3, round(dbSpan * nUnique));
    nWindow = min(nWindow, nUnique);

    if mod(nWindow,2) == 0 && nWindow < nUnique
        nWindow = nWindow + 1;
    end

    halfWindow = floor(nWindow / 2);
    index = (1:nUnique)';
    lowerIndex = max(1, index - halfWindow);
    upperIndex = min(nUnique, index + halfWindow);

    % 加权局部线性回归需要：
    %   sum(w), sum(w*x), sum(w*y), sum(w*x^2), sum(w*x*y)
    wx  = groupCount .* predictorUnique;
    wy  = groupSumY;
    wxx = groupCount .* predictorUnique.^2;
    wxy = predictorUnique .* groupSumY;

    cW   = [0; cumsum(groupCount)];
    cWX  = [0; cumsum(wx)];
    cWY  = [0; cumsum(wy)];
    cWXX = [0; cumsum(wxx)];
    cWXY = [0; cumsum(wxy)];

    SW   = cW(upperIndex+1)   - cW(lowerIndex);
    SX   = cWX(upperIndex+1)  - cWX(lowerIndex);
    SY   = cWY(upperIndex+1)  - cWY(lowerIndex);
    SXX  = cWXX(upperIndex+1) - cWXX(lowerIndex);
    SXY  = cWXY(upperIndex+1) - cWXY(lowerIndex);

    denominator = SW .* SXX - SX.^2;

    fittedUnique = zeros(nUnique,1);
    stable = abs(denominator) > ...
        1e-12 .* max(1, abs(SW .* SXX) + abs(SX.^2));

    slope = zeros(nUnique,1);
    slope(stable) = ...
        (SW(stable).*SXY(stable) - SX(stable).*SY(stable)) ./ ...
        denominator(stable);

    intercept = SY ./ SW - slope .* (SX ./ SW);
    fittedUnique = intercept + slope .* predictorUnique;

    % 当局部 predictor 方差几乎为零时，退化为局部加权均值。
    fittedUnique(~stable) = SY(~stable) ./ SW(~stable);

    fitted = fittedUnique(inverseIndex);
end


function z = localStandardize(x)
%LOCALSTANDARDIZE 中心化并缩放到样本标准差为 1。

    x = x(:);
    xCentered = x - mean(x);
    s = std(xCentered, 0);

    if ~isfinite(s) || s <= 1e-12
        z = zeros(size(x));
    else
        z = xCentered / s;
    end
end


function r = localSafePearson(x, y)
%LOCALSAFEPEARSON 在常数向量时返回 NaN，避免 corr 报警或异常。

    if std(x,0) <= 1e-12 || std(y,0) <= 1e-12
        r = NaN;
    else
        r = corr(x, y, 'Type', 'Pearson');
    end
end


function r = localSafeSpearman(x, y)
%LOCALSAFESPEARMAN 在常数向量时返回 NaN。

    if std(x,0) <= 1e-12 || std(y,0) <= 1e-12
        r = NaN;
    else
        r = corr(x, y, 'Type', 'Spearman');
    end
end
