function rho=entrain_index(phase_difference,bins)
xval=linspace(0,2*pi,bins+1);
xval=xval(1:end-1);
[n,~]=hist(phase_difference,xval);
p=n./(sum(n));
temp_p=p;
temp_p(temp_p==0)=1;
log_p=log(temp_p);
s=-sum(p.*log_p);
rho=1-s/log(bins);