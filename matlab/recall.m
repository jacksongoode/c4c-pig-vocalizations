function y = recall(M)
  y = diag(M) ./ sum(M,1)';
end