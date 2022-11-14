function index=kuramoto(pd)
index=0+0*1i;
for n=1:length(pd)
    index=index+exp(pd(n)*1i);
    disp(index)
end
index=abs(index/length(pd));
end