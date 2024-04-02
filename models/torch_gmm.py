from torch.nn import Module
from torch import nn

class GMM(Module):
    def __init__(self, num_gauss, num_bpe, fea_dim=1024):
        super().__init__()
        
        self.mu = nn.Parameter(torch.rand((num_bpe, num_gauss, fea_dim)), requires_grad=True)
        self.log_cov = nn.Parameter(torch.log(torch.ones((num_bpe, num_gauss, fea_dim))), requires_grad=True)
        self.log_pi = nn.Parameter(torch.log(torch.ones((num_bpe, num_gauss)) / num_gauss), requires_grad=True)
        
    def forward(self, X):
        return compute_gmm_pdf_batch(X, self.mu, self.log_cov, self.log_pi)


def compute_gmm_pdf_batch(seq, mu, log_cov, log_pi):
    """
    input: seq: BxTxF
    mu: KxLxF
    cov: KxLxF
    log_pi: KxL
    out: BxTx1 - likelihoods
    """
    
    D = mu.shape[2] # fea dimension
    L = mu.shape[1] # num Gaussians
    K = mu.shape[0] # num BPE
    
    C = -torch.ones((K, L), device=mu.device) * D/2 * torch.log(torch.tensor([2*np.pi], device=mu.device)) - 0.5*torch.sum(log_cov, -1)
    
    # We need BxTxKxLxF
    # Then eval gauss likelihoods - BxTxKxL
    # Then sum across weighted gaussians to obtain BxTxK tensor
    
    # unnormalized log likelihood - BxTxtKxLxF
    seq_centered = (seq.unsqueeze(2).unsqueeze(3) - mu.unsqueeze(0).unsqueeze(0))
    ULL = 0.5 * torch.sum(seq_centered * seq_centered / torch.exp(log_cov).unsqueeze(0).unsqueeze(0), -1)
    return torch.logsumexp(C.unsqueeze(0).unsqueeze(0) - ULL + (log_pi.unsqueeze(0).unsqueeze(0) - torch.logsumexp(log_pi.unsqueeze(0).unsqueeze(0), dim=-1).unsqueeze(-1)), -1)

