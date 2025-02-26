function y = precision(M)
  y = diag(M) ./ sum(M,2);
end