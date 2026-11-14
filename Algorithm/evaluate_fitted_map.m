function newz=evaluate_fitted_map(x,z,newx)
%EVALUATE_FITTED_MAP Frozen feature map for ACE, MOS and COMET.
% Inside the training range: piecewise-linear interpolation.
% Outside the training range: the corresponding endpoint score.
% Clamping can create ties outside the fitted range; it does not change
% the fitted nodes or impose positivity on the transformed scores.
    x=x(:);shape=size(newx);query=newx(:);
    if isvector(z)&&numel(z)==numel(x),z=z(:);end
    assert(size(z,1)==numel(x)&&~isempty(x));
    assert(all(isfinite(x))&&all(isfinite(z),'all')&&~any(isnan(query)));
    [knots,first,g]=unique(x,'sorted');values=z(first,:);
    assert(all(z==values(g,:),'all'),'Equal inputs must have equal fitted scores.');
    if numel(knots)==1
        newz=repmat(values,numel(query),1);
    else
        query=min(max(query,knots(1)),knots(end));
        index=discretize(query,[-Inf;knots(2:end-1);Inf]);
        t=(query-knots(index))./(knots(index+1)-knots(index));
        left=values(index,:);right=values(index+1,:);
        newz=left+t.*(right-left);
        newz=min(max(newz,min(left,right)),max(left,right));
        newz(query==knots(1),:)=repmat(values(1,:),sum(query==knots(1)),1);
        newz(query==knots(end),:)=repmat(values(end,:),sum(query==knots(end)),1);
    end
    if size(z,2)==1,newz=reshape(newz,shape);end
end
