classdef VirtualSample < handle
    % 用于本案例的虚拟样本生成方法，输入输出均为矩阵类型
    
    properties
        
        data_new;  %新生成的数据
        num_total; %原始样本+虚拟样本的数量
        
        % 输入参数
        vir_num;  % 生成的虚拟样本的数量
        affi;     % 隶属度函数值
        %X_dim;    % X的维度
        %method;   % 生成样本的方法{'Diff','Iter'}，一般选用'Iter'（'Diff'方法没有写，参考原Python程序）
        %xls_name; % 原始数据文件名：数据格式[X, y]
        rand_num; % 样本打乱的随机数
        
        data_ori;        % 提取出的原始数据表格
    end
    
    methods
        function obj = VirtualSample(varargin)
            % 输入：{'vir_num',120, 'affi', 0.99, 'X_dim', 12, 'method', 'Iter', 'xls_name', 'Data.xlsx','rand_num', 44}
            p = inputParser;

            default_vir_num = 100;
            default_affi = 0.99;

            default_rand_num = 15;
            
            p.addRequired('data_ori',@isnumeric);
            p.addOptional('vir_num',default_vir_num,@isnumeric);
            p.addOptional('affi',default_affi,@isnumeric);
            p.addOptional('rand_num',default_rand_num,@isnumeric);
            
            p.parse(varargin{:}); % 解析函数输入
            
            obj.data_ori = p.Results.data_ori;
            obj.vir_num = p.Results.vir_num;
            obj.affi = p.Results.affi;
            obj.rand_num = p.Results.rand_num;
            
            %obj.T = readtable(obj.xls_name,'VariableNamingRule','preserve'); % 读取表格且保留变量名称的标志
            
            obj = obj.sample_generate();
        end
        
        function obj = sample_generate(obj)
            % 'Iter'样本生成方法
            data = obj.data_ori;
            obj.num_total = [size(data,1)];
            while obj.num_total(end) < obj.vir_num+size(obj.data_ori,1) 
                mean_values = mean(data);
                var_values = var(data);
                
                for i = 1:size(data,2)
                    num_l(i) = length(find(data(:,i)>mean_values(:,i)));
                end
                Skew_L = num_l/obj.num_total(end);
                Skew_U = 1-Skew_L;
                
                %% 样本生成
                affi_seq = obj.affi.*ones(1,size(data,2));
                
                % 随着迭代次数，逐渐缩小隶属度值，以减慢发散。（可以将0.001替换为随样本数量的自适应调整策略）
                affi_seq = affi_seq+0.0005;               
                
                %vir_sample_l = data-Skew_L.*sqrt(-2*var_values.*log(affi_seq));
                vir_sample_u = data+Skew_U.*sqrt(-2*var_values.*log(affi_seq));
                
                %% 剔除不满足条件的样本
%                 for i = 1:size(data,2)
%                     vir_sample_l(vir_sample_l(:,i)<0|vir_sample_l(:,i)>1,:) = [];
%                     vir_sample_u(vir_sample_u(:,i)<0|vir_sample_u(:,i)>1,:) = [];
%                 end
                data = [data; vir_sample_u];
                %data = [data; vir_sample_l; vir_sample_u];
                obj.num_total = [obj.num_total, size(data,1)];
            end
            % 从最后一次生成的数据中挑选
            if size(obj.num_total,2)~=1
                num_left = obj.vir_num+size(obj.data_ori,1)-obj.num_total(end-1); % 剩余待补充的样本数量
                unselect = obj.num_total(end-1)+randperm(obj.num_total(end)-obj.num_total(end-1)); % 生成最后一次生成的样本的随机序列
                unselect(1:num_left) = []; % 提出待补充的样本数
                data(unselect,:) = [];
            end
            shuffle = randperm(size(data,1));
            
            obj.data_new = data(shuffle,:);
        end
            

    end
end

