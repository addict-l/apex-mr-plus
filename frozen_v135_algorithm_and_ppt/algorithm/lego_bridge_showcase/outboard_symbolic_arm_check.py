"""Read-only symbolic arm check for the locked Mode B input."""
import json
import symbolic_modes

def inspect(path, plan):
    assert path.name == "plan.json"
    deck=[{"id":row["id"],"skill":row["skill"],"arm":row.get("arm")}
          for row in plan["actions"] if row["part"]=="Bridge_deck_12x4"]
    print(json.dumps({"published":False,"path_intercepted":str(path),
                      "input_sha256":plan["input_sha256"],"deck_steps":deck,
                      "symbolic_count":len(plan["actions"]),"status":plan["status"]}))

symbolic_modes.publish=inspect
raise SystemExit(symbolic_modes.solve("mode_b"))