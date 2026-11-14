function [matZ, stats] = transform_MOS(matX, cvY)
% Held-out feature scores: apply_map(x_train,z_train,x_new).
% Uses linear interpolation inside training support and endpoint clamping outside.
% Clamping may create held-out boundary ties; fitted training nodes are unchanged.

% MOS 对每个代谢特征执行 Monotone Optimal Scaling。
%
%   [matZ, STATS] = transform_MOS(matX, cvY)
%
%   对 matX 中每一列代谢物 x_j 独立寻找一个关于 x_j 的非降
%   optimal-scaling score q_j(x_j)，使原始连续表型 cvY 可以通过
%
%       cvY = beta0 + beta1 * q_j(x_j) + error
%
%   尽可能被单变量线性回归解释。表型 cvY 本身保持不变。
%
%   本实现把连续代谢特征的排序非重复取值视为有序水平，并使用加权
%   isotonic regression / Pool-Adjacent-Violators Algorithm (PAVA)
%   求解单调最小二乘 optimal quantification。程序同时评价“表型随 x
%   单调增加”和“表型随 x 单调减少”两个方向，并选择残差平方和较小者。
%
%   与 COMET 的关键区别
%   -------------------
%   1. MOS 不包含原始丰度扰动惩罚。
%   2. MOS 只要求 q_j(x) 非降，而不是严格递增。
%   3. PAVA 会将违反单调性的相邻水平合并，因此 MOS 可以产生 ties，
%      变换前后的 Spearman 相关通常不一定等于 1。
%   4. MOS 保持原始 cvY 不变，因此比标准 ACE 更接近 COMET 的
%      fixed-phenotype 比较目标。
%
%   输入参数
%   --------
%   matX
%       n×p 数据矩阵，其中每一行代表一个样本，每一列代表一个代谢物。
%
%   cvY
%       n×1 连续表型向量。
%
%   输出参数
%   --------
%   matZ
%       n×p 的 Monotone Optimal Scaling 变换结果。
%       为使输出尺度与原始数据便于比较，每个最优分数最终只经过正仿射
%       重标度，使其均值和标准差与原始 x_j 相同。该操作不改变相关系数、
%       线性回归 R^2 或单调关系。
%
%   stats
%       保存各代谢物结果的结构体：
%
%         rvRawPCC
%             原始 x_j 与 cvY 的 Pearson 相关。
%
%         rvPCC
%             MOS 变换后 z_j 与 cvY 的 Pearson 相关。
%
%         rvR2
%             cvY ~ 1 + z_j 的单变量线性回归 R^2。
%
%         rvSpearman
%             z_j 与原始 x_j 的 Spearman 相关。
%
%         rvDeltaNorm
%             ||z_j-x_j||_2。
%
%         rvRelativeNorm
%             ||z_j-x_j||_2^2 /
%             ||x_j-mean(x_j)||_2^2。
%
%         rvDirection
%             +1 表示选择了随 x 单调增加的表型拟合；
%             -1 表示选择了随 x 单调减少的表型拟合。
%             无论方向如何，输出 q_j(x) 始终是非降函数。
%
%         rvSSE
%             所选单调回归方向的训练残差平方和。
%
%         rvNumberOfBlocks
%             PAVA 最终保留的不同单调分数块数量。
%
%         rvPoolFraction
%             1 - nBlock/nUnique。数值越大表示越多相邻 x 水平被合并。
%
%         note
%             当前特征处理状态。
%
%         arrKnotX / arrKnotZ
%             原始排序非重复 x 节点以及对应的 MOS 变换节点。
%
%   数学原理
%   --------
%   令 x 的排序非重复水平为 u_1 < ... < u_m，相应样本数为 w_k，
%   组内表型均值为 ybar_k。增加方向求解
%
%       min sum_k w_k * (ybar_k - a_k)^2
%
%       subject to  a_1 <= a_2 <= ... <= a_m。
%
%   减少方向等价于对 -ybar 做同样的 isotonic regression。两者均由
%   weighted PAVA 精确求解。若减少方向更优，则输出 q(x)=-a_dec(x)，
%   这样 q(x) 仍为非降，而后续线性模型通过负回归系数表示下降趋势。
%
%   重要说明
%   --------
%   MOS 是监督式变换，可能在空特征上产生乐观的训练集线性化。
%   正式方法学比较应同时报告 permutation null 或样本外评价。
%
%   本实现假定 matX 和 cvY 已经是数值型有限值，并且样本一一对应。
%
%   调用示例
%   --------
%   [matZ, stats] = transform_MOS(matX, cvY);

    cvY = cvY(:);
    [~, nMetabolite] = size(matX);

    matZ = matX;

    stats.transformed      = false(1, nMetabolite);
    stats.rvRawPCC         = nan(1, nMetabolite);
    stats.rvPCC            = nan(1, nMetabolite);
    stats.rvR2             = nan(1, nMetabolite);
    stats.rvSpearman       = nan(1, nMetabolite);
    stats.rvDeltaNorm      = zeros(1, nMetabolite);
    stats.rvRelativeNorm   = zeros(1, nMetabolite);
    stats.rvDirection      = zeros(1, nMetabolite);
    stats.rvSSE            = nan(1, nMetabolite);
    stats.rvNumberOfBlocks = zeros(1, nMetabolite);
    stats.rvPoolFraction   = nan(1, nMetabolite);
    stats.note             = strings(1, nMetabolite);
    stats.arrKnotX         = cell(1, nMetabolite);
    stats.arrKnotZ         = cell(1, nMetabolite);

    yVariation = sum((cvY - mean(cvY)).^2);

    for j = 1:nMetabolite
        x = matX(:,j);
        [xUnique, ~, inverseIndex] = unique(x, 'sorted');
        nUnique = numel(xUnique);

        stats.rvRawPCC(j) = localSafePearson(x, cvY);

        if nUnique <= 1 || std(x,0) <= eps(max(1,abs(mean(x))))
            matZ(:,j) = x;
            stats.rvPCC(j) = stats.rvRawPCC(j);
            stats.rvSpearman(j) = localSafeSpearman(x, x);
            stats.rvDirection(j) = 0;
            stats.rvNumberOfBlocks(j) = nUnique;
            stats.rvPoolFraction(j) = 0;
            stats.note(j) = "代谢物为常数，未执行 MOS 变换";
            stats.arrKnotX{j} = xUnique;
            stats.arrKnotZ{j} = xUnique;
            continue;
        end

        % 每个非重复 x 水平的样本数与平均表型。
        groupCount = accumarray(inverseIndex, 1, [nUnique,1], @sum);
        groupSumY = accumarray(inverseIndex, cvY, [nUnique,1], @sum);
        groupMeanY = groupSumY ./ groupCount;

        % 增加方向：Y 随 x 非降。
        [fitIncreasing, nBlockIncreasing] = ...
            localWeightedPAVA(groupMeanY, groupCount);
        predIncreasing = fitIncreasing(inverseIndex);
        sseIncreasing = sum((cvY - predIncreasing).^2);

        % 减少方向：Y 随 x 非升。
        % 对 -Y 做 isotonic regression，再翻回即可得到非升 Y 拟合。
        [fitNegativeY, nBlockDecreasing] = ...
            localWeightedPAVA(-groupMeanY, groupCount);
        fitDecreasing = -fitNegativeY;
        predDecreasing = fitDecreasing(inverseIndex);
        sseDecreasing = sum((cvY - predDecreasing).^2);

        % 选择训练 SSE 更小的方向。若数值完全相同，则用原始相关方向打破平局。
        toleranceSSE = 1e-12 * max(1, min(sseIncreasing, sseDecreasing));

        if sseIncreasing < sseDecreasing - toleranceSSE
            selectedDirection = 1;
        elseif sseDecreasing < sseIncreasing - toleranceSSE
            selectedDirection = -1;
        else
            rawPCC = stats.rvRawPCC(j);
            if isfinite(rawPCC) && rawPCC < 0
                selectedDirection = -1;
            else
                selectedDirection = 1;
            end
        end

        if selectedDirection == 1
            % fitIncreasing 自身随 x 非降，因此直接作为 optimal score。
            scoreUnique = fitIncreasing;
            selectedSSE = sseIncreasing;
            selectedBlocks = nBlockIncreasing;
        else
            % fitDecreasing 随 x 非升。取负后得到随 x 非降的 q(x)。
            scoreUnique = -fitDecreasing;
            selectedSSE = sseDecreasing;
            selectedBlocks = nBlockDecreasing;
        end

        score = scoreUnique(inverseIndex);

        % 若最优单调分数完全退化为常数，则保留原始 x。
        if std(score,0) <= 1e-12 || any(~isfinite(score))
            z = x;
            zUnique = xUnique;
            stats.rvDirection(j) = selectedDirection;
            rawPCC = stats.rvRawPCC(j);
            if isfinite(rawPCC)
                stats.rvSSE(j) = yVariation * max(0, 1 - rawPCC^2);
            else
                stats.rvSSE(j) = yVariation;
            end
            stats.rvNumberOfBlocks(j) = 1;
            stats.rvPoolFraction(j) = 1 - 1/nUnique;
            stats.note(j) = "MOS 最优分数退化为常数，保留原始数据";
        else
            % 仅用正仿射变换匹配原始 x 的均值和标准差。
            scoreStandard = (score - mean(score)) / std(score,0);
            z = mean(x) + std(x,0) * scoreStandard;

            scoreUniqueMean = mean(score);
            scoreUniqueStd = std(score,0);
            zUnique = mean(x) + std(x,0) * ...
                ((scoreUnique - scoreUniqueMean) / scoreUniqueStd);

            stats.transformed(j) = true;
            stats.rvDirection(j) = selectedDirection;
            stats.rvSSE(j) = selectedSSE;
            stats.rvNumberOfBlocks(j) = selectedBlocks;
            stats.rvPoolFraction(j) = 1 - selectedBlocks/nUnique;
            stats.note(j) = "已执行 Monotone Optimal Scaling";
        end

        matZ(:,j) = z;

        stats.rvPCC(j) = localSafePearson(z, cvY);
        stats.rvSpearman(j) = localSafeSpearman(z, x);
        stats.rvDeltaNorm(j) = norm(z - x);

        xVariation = sum((x - mean(x)).^2);
        if xVariation > 0
            stats.rvRelativeNorm(j) = sum((z - x).^2) / xVariation;
        else
            stats.rvRelativeNorm(j) = 0;
        end

        if yVariation > 0 && isfinite(stats.rvPCC(j))
            stats.rvR2(j) = stats.rvPCC(j)^2;
        else
            stats.rvR2(j) = NaN;
        end

        stats.arrKnotX{j} = xUnique;
        stats.arrKnotZ{j} = zUnique;
    end
end


function [fit, nBlock] = localWeightedPAVA(y, w)
%LOCALWEIGHTEDPAVA
%   加权 Pool-Adjacent-Violators Algorithm。
%
%   求解
%
%       min sum_i w_i * (y_i - a_i)^2
%
%       subject to a_1 <= a_2 <= ... <= a_m。
%
%   返回每个原始有序水平对应的 isotonic fitted value。

    y = y(:);
    w = w(:);
    m = numel(y);

    blockLevel = zeros(m,1);
    blockWeight = zeros(m,1);
    blockStart = zeros(m,1);
    blockEnd = zeros(m,1);

    nBlock = 0;

    for i = 1:m
        nBlock = nBlock + 1;
        blockLevel(nBlock) = y(i);
        blockWeight(nBlock) = w(i);
        blockStart(nBlock) = i;
        blockEnd(nBlock) = i;

        % 只要前一块水平高于后一块，就合并相邻违反者。
        while nBlock > 1 && blockLevel(nBlock-1) > blockLevel(nBlock)
            combinedWeight = blockWeight(nBlock-1) + blockWeight(nBlock);
            combinedLevel = ...
                (blockWeight(nBlock-1) * blockLevel(nBlock-1) + ...
                 blockWeight(nBlock)   * blockLevel(nBlock)) / combinedWeight;

            blockLevel(nBlock-1) = combinedLevel;
            blockWeight(nBlock-1) = combinedWeight;
            blockEnd(nBlock-1) = blockEnd(nBlock);

            nBlock = nBlock - 1;
        end
    end

    fit = zeros(m,1);
    for b = 1:nBlock
        fit(blockStart(b):blockEnd(b)) = blockLevel(b);
    end
end


function r = localSafePearson(x, y)
%LOCALSAFEPEARSON 在常数向量时返回 NaN。

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
