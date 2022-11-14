figure
tmp=period_vector27_both;
frequency=zeros(1,14);
for i=1:length(tmp)
    for j=1:14
        if tmp(i)>10+(j-1)*3 && tmp(i)<10+j*3
            frequency(j)=frequency(j)+1;
            break
        end
    end
end
frequency=frequency/sum(frequency);
bar(frequency)
xticks([1,14]);xticklabels([12,51])
yticks([])
axis([-inf,inf,0,0.3])