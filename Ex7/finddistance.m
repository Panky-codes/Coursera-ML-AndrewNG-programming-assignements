function dist =  finddistance(X_i,mu_i)



dis = 0;
for i = 1 : size(X_i,2)
dis = dis + (X_i(1,i) - mu_i(1,i))^2;
endfor 
dist = dis;
