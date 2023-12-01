# optimizer
optimizer = dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=6400)
# evaluation = dict(interval=3200, metric=['precision', 'recall'])
evaluation = dict(interval=3200, metric='mIoU')


# # optimizer
# optimizer = dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=0.0005)
# optimizer_config = dict()
# # learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
# # runtime settings
# runner = dict(type='IterBasedRunner', max_iters=50000)
# checkpoint_config = dict(by_epoch=False, interval=5000)
# # evaluation = dict(interval=3200, metric=['precision', 'recall'])
# evaluation = dict(interval=2500, metric='mIoU')
