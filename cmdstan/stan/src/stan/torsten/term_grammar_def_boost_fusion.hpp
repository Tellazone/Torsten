#ifndef STAN_LANG_TORSTEN_GRAMMARS_TERM_GRAMMAR_DEF_BOOST_FUSION_HPP
#define STAN_LANG_TORSTEN_GRAMMARS_TERM_GRAMMAR_DEF_BOOST_FUSION_HPP

BOOST_FUSION_ADAPT_STRUCT(stan::lang::univariate_integral_control,
                          (std::string, integration_function_name_)
                          (std::string, system_function_name_)
                          (stan::lang::expression, t0_)
                          (stan::lang::expression, t1_)
                          (stan::lang::expression, theta_)
                          (stan::lang::expression, x_r_)
                          (stan::lang::expression, x_i_) )

BOOST_FUSION_ADAPT_STRUCT(stan::lang::generalOdeModel_control,
                          (std::string, integration_function_name_)
                          (std::string, system_function_name_)
                          (stan::lang::expression, nCmt_)
                          (stan::lang::expression, time_)
                          (stan::lang::expression, amt_)
                          (stan::lang::expression, rate_)
                          (stan::lang::expression, ii_)
                          (stan::lang::expression, evid_)
                          (stan::lang::expression, cmt_)
                          (stan::lang::expression, addl_)
                          (stan::lang::expression, ss_)
                          (stan::lang::expression, pMatrix_)
                          (stan::lang::expression, biovar_)
                          (stan::lang::expression, tlag_)
                          (stan::lang::expression, rel_tol_)
                          (stan::lang::expression, abs_tol_)
                          (stan::lang::expression, max_num_steps_) )

BOOST_FUSION_ADAPT_STRUCT(stan::lang::generalOdeModel,
                          (std::string, integration_function_name_)
                          (std::string, system_function_name_)
                          (stan::lang::expression, nCmt_)
                          (stan::lang::expression, time_)
                          (stan::lang::expression, amt_)
                          (stan::lang::expression, rate_)
                          (stan::lang::expression, ii_)
                          (stan::lang::expression, evid_)
                          (stan::lang::expression, cmt_)
                          (stan::lang::expression, addl_)
                          (stan::lang::expression, ss_)
                          (stan::lang::expression, pMatrix_)
                          (stan::lang::expression, biovar_)
                          (stan::lang::expression, tlag_) )

#endif
