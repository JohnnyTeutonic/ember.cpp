#ifndef EMBER_DECODE_PICK_POLICY_H
#define EMBER_DECODE_PICK_POLICY_H

#include <cstring>
#include <stdexcept>

namespace ember {
// Experimental selector integration only. Never silently enable deterministic
// arithmetic or turn on compact picking; the caller must request both itself.
inline int decode_pick_partitions(const char* value, bool deterministic) {
  if (!value || !*value || std::strcmp(value, "0") == 0) return 0;
  if (std::strcmp(value, "4") != 0)
    throw std::invalid_argument("EMBER_DECODE_PICK_PARTS must be 0 or 4");
  if (!deterministic)
    throw std::invalid_argument("EMBER_DECODE_PICK_PARTS=4 requires EMBER_DETERMINISTIC=1; arithmetic was not changed");
  return 4;
}

inline bool decode_pick_graph_matches(const void* cached_kv, const void* requested_kv,
                                      bool cached_pick, bool requested_pick,
                                      int cached_parts, int requested_parts) {
  return cached_kv == requested_kv && cached_pick == requested_pick &&
         (!requested_pick || cached_parts == requested_parts);
}
}  // namespace ember
#endif
