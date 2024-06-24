classdef VirtualSample < handle
    %虚拟样本生成类
    
    properties
        
        data_new;  %新生成的数据
        num_total; %原始样本+虚拟样本的数量
        
        % 输入参数
        vir_num;  % 生成的虚拟样本的数量
        affi;     % 隶属度函数值
        X_dim;    % X的维度
        method;   % 生成样本的方法{'Diff','Iter'}，一般选用'Iter'（'Diff'方法没有写，参考原Python程序）
        xls_name; % 原始数据文件名：数据格式[X, y]
        rand_num; % 样本打乱的随机数
        
        T;        % 提取出的原始数据表格
    end
    
    methods
        function obj = VirtualSample(varargin)
            % 输入：{'vir_num',120, 'affi', 0.99, 'X_dim', 12, 'method', 'Iter', 'xls_name', 'Data.xlsx','rand_num', 44}
            p = inputParser;
            
            default_method = 'Iter';
            valid_method = {'Diff','Iter'};
            checkdata_type = @(x) any(validatestring(x,valid_method));
            
            default_xls_name = 'Data.xlsx';
            
            default_vir_num = 300;
            default_affi = 0.99;
            default_X_dim = 12;

            default_rand_num = 15;
            
            p.addOptional('vir_num',default_vir_num,@isnumeric);
            p.addOptional('affi',default_affi,@isnumeric);
            p.addOptional('X_dim',default_X_dim,@isnumeric);
            p.addOptional('method',default_method,checkdata_type);
            p.addOptional('xls_name',default_xls_name);
            p.addOptional('rand_num',default_rand_num,@isnumeric);
            
            p.parse(varargin{:}); % 解析函数输入
            
            obj.vir_num = p.Results.vir_num;
            obj.affi = p.Results.affi;
            obj.X_dim = p.Results.X_dim;
            obj.method = p.Results.method;
            obj.xls_name = p.Results.xls_name;
            obj.rand_num = p.Results.rand_num;
            
            obj.T = readtable(obj.xls_name,'VariableNamingRule','preserve'); % 读取表格且保留变量名称的标志
            
            obj = obj.sample_generate();
        end
        
        function obj = sample_generate(obj)
            % 'Iter'样本生成方法
            data = obj.T.Variables;
            
            num_sample = size(obj.T,1);
            obj.num_total = num_sample;
            while obj.num_total < obj.vir_num+size(obj.T,1) 
                mean_values = mean(data);
                var_values = var(data);
                
                for i = 1:size(data,2)
                    num_l(i) = length(find(data(:,i)>mean_values(:,i)));
                end
                Skew_L = num_l/num_sample;
                Skew_U = 1-Skew_L;
                
                %% 样本生成
                affi_seq = obj.affi.*ones(1,size(data,2));
                %vir_sample_l = data-Skew_L.*sqrt(-2*var_values.*log(affi_seq));
                vir_sample_u = data+Skew_U.*sqrt(-2*var_values.*log(affi_seq));
                
                %% 剔除不满足条件的样本
                for i = 1:size(data,2)
                    %vir_sample_l(vir_sample_l(:,i)<0|vir_sample_l(:,i)>1,:) = [];
                    vir_sample_u(vir_sample_u(:,i)<0|vir_sample_u(:,i)>1,:) = [];
                end
                
                data = [data;vir_sample_u];
                obj.num_total = obj.num_total+size(data,1);
            end
            obj.data_new = data;

        end
            

    end
end

