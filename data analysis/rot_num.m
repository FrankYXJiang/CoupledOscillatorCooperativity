function [omega_matrix,omega,omega_std]=rot_num(data,period,t_interval)
%this script calculates the rotation number of a certain experiment condition
%make sure that each row(not column) of the data represents one cell!
%事实证明这算法不好用。。。use find_T.m instead

n=ceil((size(data,2)-1)*t_interval/period);%how many [whole period] are there in the time trace
omega_matrix=zeros(size(data,1),n);

for i=1:size(data,1)

        [~,peakpos]=find_peaks(data(i,:),0.1);
        if isempty(peakpos)
        else
            for j=1:length(peakpos)
                omega_matrix(i,ceil((peakpos(j)-1)*t_interval/period))=omega_matrix(i,ceil((peakpos(j)-1)*t_interval/period))+1;
            end
        end

end
omega=mean(reshape(omega_matrix,1,[]));
omega_std=std(reshape(omega_matrix,1,[]));
end