% increase function
%v:nonlinear factor
%i_or_d: 1->increase function   0->decrease function
%delta: calculated delta_G from back propagation
function deltaG_actual=nonlinearG(Gmax,Gmin,v,G,delta)

if v~=0
    deltaG_actual=zeros(size(G));
    G1=(Gmax-Gmin)/(1-exp(-v));
    deltaG=delta/(Gmax-Gmin);%normalized G
    
    G_pos=deltaG>0;%G is going to reduce
    G_neg=deltaG<0;%G is going to increase
    
    G_tmp=zeros(size(G));
    G_tmp(G_neg)=(1-exp(-v.*deltaG(G_neg)));
    deltaG_actual_neg=(G1+Gmin-G).*G_tmp;
    
    G_tmp=zeros(size(G));
    G_tmp(G_pos)=(1-exp(v.*deltaG(G_pos)));
    deltaG_actual_pos=-(G+G1-Gmax).*G_tmp;
    
    deltaG_actual=deltaG_actual_neg+deltaG_actual_pos;
        
else
    G_i=Gmin+(Gmax-Gmin)*P;
    G_d=G_i;
end

end