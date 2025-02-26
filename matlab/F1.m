function [p_vect,r_vect,f_vect,weighted_p,weighted_r,weighted_f] = F1(M,valLabelCount)
  p_vect=precision(M);
  r_vect=recall(M);
  f_vect=2.*p_vect.*r_vect./(p_vect+r_vect);
  
  weights = valLabelCount{:,2}./sum(valLabelCount{:,2});
  weighted_p = sum(p_vect.*weights);
  weighted_r = sum(r_vect.*weights);
  weighted_f = sum(f_vect.*weights);
end