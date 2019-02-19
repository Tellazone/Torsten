#ifndef STAN_LANG_AST_NODE_POP_PK_GENERALODEMODEL_DEF_HPP
#define STAN_LANG_AST_NODE_POP_PK_GENERALODEMODEL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    pop_pk_generalOdeModel::pop_pk_generalOdeModel() { }

    pop_pk_generalOdeModel::pop_pk_generalOdeModel (const std::string& integration_function_name,
                                                                              const std::string& system_function_name,
                                                                              const expression& nCmt,
                                                                              const expression& len,
                                                                              const expression& time,
                                                                              const expression& amt,
                                                                              const expression& rate,
                                                                              const expression& ii,
                                                                              const expression& evid,
                                                                              const expression& cmt,
                                                                              const expression& addl,
                                                                              const expression& ss,
                                                                              const expression& len_pMatrix,
                                                                              const expression& pMatrix,
                                                                              const expression& len_biovar,
                                                                              const expression& biovar,
                                                                              const expression& len_tlag,
                                                                              const expression& tlag)
    : integration_function_name_(integration_function_name),
      system_function_name_(system_function_name),
      nCmt_(nCmt),
      len_(len),
      time_(time), amt_(amt), rate_(rate), ii_(ii),
      evid_(evid), cmt_(cmt), addl_(addl), ss_(ss),
      len_pMatrix_(len_pMatrix), pMatrix_(pMatrix),
      len_biovar_(len_biovar), biovar_(biovar),
      len_tlag_(len_tlag), tlag_(tlag)
    {}
  }
}
#endif
