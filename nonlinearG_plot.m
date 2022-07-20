%G_i: increase pulse
%G_d: decrease pulse
%v: nonlinear factor
%G: the current conductance
function [G_i,G_d]=nonlinearG_plot(Gmax,Gmin,v,P)
if ~v==0
    G1=(Gmax-Gmin)/(1-exp(-v));
    G_i=G1*(1-exp(-v*P))+Gmin;
    G_d=Gmax-G1*(1-exp(-v*(1-P)));
else
    G_i=(Gmax-Gmin)*P;
    G_d=G_i;
end