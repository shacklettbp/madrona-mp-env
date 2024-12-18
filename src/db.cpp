#include "db.hpp"

namespace madronaMPEnv {

void execSQL(sqlite3 *db, const char *sql)
{
  char * err_msg;
  int res = sqlite3_exec(db, sql, nullptr, nullptr, &err_msg);

  if (res == SQLITE_OK) [[likely]] {
    return;
  }

  FATAL("SQL error executing '%s': %s", sql, err_msg);
}

void execResetStmt(sqlite3 *db, sqlite3_stmt *stmt)
{
  if (sqlite3_step(stmt) != SQLITE_DONE) {
    FATAL("Failed to execute statement: %s", sqlite3_errmsg(db));
  }

  REQ_SQL(db, sqlite3_reset(stmt));
}

}
