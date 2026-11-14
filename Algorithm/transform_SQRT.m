function [matZ, dbLamda, stats] = transform_SQRT(matX, cvGroup)
%TRANSFORM_SQRT 普通平方根；要求有限且非负。
%   [matZ, dbLamda, stats] = transform_SQRT(matX, cvGroup)
%   matX为样本×代谢物的实数数值矩阵。cvGroup可省略，仅用于统一接口，
%   不参与计算。本函数不拟合参数，不缩放数据、不插补、不添加伪计数。
%   含非有限值或超出定义域的列保留原值，stats.transformed为false。
%   dbLamda是固定幂参数标签（未变换列为NaN），不是拟合结果。
%   正态性得分字段保留为NaN，以兼容transform_ABC的输出结构。
%   输出为sqrt(x)，不是Box-Cox在lambda=0.5时的2*(sqrt(x)-1)。

    validateattributes(matX, {'numeric'}, {'real', '2d'}, mfilename, 'matX');
    matZ = double(matX);
    nMetabolite = size(matX, 2);
    dbLamda = nan(1, nMetabolite);
    stats.transformed = false(1, nMetabolite);
    stats.note = strings(1, nMetabolite);
    stats.originalScore = nan(1, nMetabolite);
    stats.selectedScore = nan(1, nMetabolite);

    for j = 1:nMetabolite
        x = double(matX(:, j));
        if isempty(x) || any(~isfinite(x)) || any(x < 0)
            stats.note(j) = "数据为空、含非有限值或超出定义域，保留原值";
            continue;
        end
        matZ(:, j) = sqrt(x);
        dbLamda(j) = 0.5;
        stats.transformed(j) = true;
        stats.note(j) = "已执行SQRT变换";
    end
end
