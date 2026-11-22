use std::collections::HashMap;
use std::sync::Arc;

pub trait Handler<T, R> {
    fn handle(&self, input: T) -> Result<R, String>;
}

pub struct ServiceRegistry<K: std::hash::Hash + Eq, V> {
    items: HashMap<K, Arc<V>>,
}

macro_rules! emit_metric {
    ($name:expr, $val:expr) => {
        record_counter($name, $val);
    };
}

impl<K: std::hash::Hash + Eq, V> ServiceRegistry<K, V> {
    pub fn new() -> Self {
        initialize_registry();
        Self {
            items: HashMap::new(),
        }
    }

    pub fn register_service(
        &mut self,
        key: K,
        service: Arc<V>,
    ) -> Result<(), String> {
        emit_metric!("service.registered", 1);
        validate_service(&key);
        self.items.insert(key, service);
        Ok(())
    }
}
