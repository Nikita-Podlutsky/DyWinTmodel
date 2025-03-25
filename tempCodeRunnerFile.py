        for name, stats in epoch_weight_stats.items():
            weight_stats_summary[name] = {
                'param_norm_mean': sum(stats['param_norms']) / len(stats['param_norms']),
                'grad_norm_mean': sum(stats['grad_norms']) / len(stats['grad_norms']),
                'grad_norm_max': max(stats['grad_norms']),
                'grad_norm_min': min(stats['grad_norms']),
            }
        weight_stats_history.append(weight_stats_summary)
        grad_norms_history.append(avg_grad_norm)