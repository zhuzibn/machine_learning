%only nonlinear case is programmed. If the synapse is linear, you should
%not reach this code
%v: nonlinear factor
%G: the current conductance
%delta: calculated delta_G from back propagation
%deltaG_actual: actual G variation due to the nonlinearity
function deltaG_actual=nonlinearG(Gmax,Gmin,v,G,delta)
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

end