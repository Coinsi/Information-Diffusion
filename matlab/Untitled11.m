T1 = readtable('Data.xlsx','VariableNamingRule','preserve');
T2 = readtable('Data2.xlsx','VariableNamingRule','preserve');

data_ori = T2.Variables;
data_c = T1.Variables;

virtual_data = VirtualSample(data_ori, 'vir_num', 400);

data_new = virtual_data.data_new;

%data_new(14,:) = []

out_index = find(data_new(:,end)>4);

data_new(out_index,:) = [];
train_num = round(0.7*size(data_new,1));

data_train = data_new(1:train_num,:);
data_test = data_new(train_num+1:end,:);