% EXAMPLE_COMET Fit a continuous-phenotype transformation and reuse its maps.
% Run from this directory. Requires Optimization and Statistics toolboxes.
rng(42,'twister');
ytrain = linspace(0,1,50)';
Xtrain = [0.2+ytrain.^2+0.02*randn(50,1), ...
          0.2+exp(-2*ytrain)+0.02*randn(50,1)];
Xtrain(2,1)=Xtrain(1,1); % An existing tie should be retained.
Xtrain(:,3)=2;           % A constant feature should be unchanged.
[Ztrain,model]=transform_COMET_continuous_stable( ...
    Xtrain,ytrain,0.5,0.01,400,10000,1e-9);
assert(all(isfinite(Ztrain),'all'));
assert(Ztrain(1,1)==Ztrain(2,1));
assert(isequal(Ztrain(:,3),Xtrain(:,3)));
for j=1:2
    assert(all(diff(model.arrKnotZ{j})>0));
    assert(abs(model.rvSpearman(j)-1)<1e-12);
    zcheck=interp1(model.arrKnotX{j},model.arrKnotZ{j},Xtrain(:,j));
    assert(max(abs(zcheck-Ztrain(:,j)))<1e-10);
end
% New abundances: same features in the same order; no phenotype required.
Xnew=[linspace(min(Xtrain(:,1)),max(Xtrain(:,1)),8)', ...
      linspace(min(Xtrain(:,2)),max(Xtrain(:,2)),8)',2*ones(8,1)];
Znew=zeros(size(Xnew));
fractionExtrapolated=zeros(1,size(Xnew,2));
for j=1:size(Xnew,2)
    xk=model.arrKnotX{j};zk=model.arrKnotZ{j};
    fractionExtrapolated(j)=mean(Xnew(:,j)<xk(1) | Xnew(:,j)>xk(end));
    if numel(xk)==1
        Znew(:,j)=Xnew(:,j);
    else
        Znew(:,j)=interp1(xk,zk,Xnew(:,j),'linear','extrap');
    end
end
assert(all(isfinite(Znew),'all'));
disp(table((1:3)',model.rvRawPCC',model.rvPCC',model.rvSpearman', ...
    'VariableNames',{'Feature','RawPearson','TransformedPearson','RankCorrelation'}));
fprintf('Example passed: finite outputs, strict knot order, preserved ties, constant-feature identity, and map reuse.\n');
